#include "gturbo/tokenizer.hpp"
#include "gturbo/format.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <filesystem>
#include <algorithm>

// Last include on purpose: check.hpp redefines assert so it survives NDEBUG, and <cassert>
// (pulled in by anything included after it) would silently define assert back to a no-op.
#include "check.hpp"

namespace fs = std::filesystem;

int main() {
    std::cout << "[TEST] Running gturbo Tokenizer unit tests...\n";

    gturbo::Tokenizer tok;

    // Test 1: A default-constructed Tokenizer holds NO vocabulary.
    // There used to be an init_default_vocab() that installed ~460 hand-written ASCII
    // pieces while reporting vocab_size() == 256000. Every decode against it produced
    // plausible-looking garbage (the "@ABCDEFGHIJ...{|}~" strings) instead of failing.
    assert(!tok.is_loaded());
    assert(tok.vocab_size() == 262144);  // Gemma 4, per the Swift reference
    assert(tok.bos_id() == 2);
    assert(tok.eos_id() == 1);
    assert(tok.pad_id() == 0);
    assert(tok.unk_id() == 3);
    assert(tok.end_of_turn_id() == 106);
    std::cout << "  [PASS] Default-constructed tokenizer is empty with Gemma 4 special IDs.\n";

    // Test 2: Stop tokens are <eos>, <turn|>, <|tool_response>.
    {
        const auto& stops = tok.stop_token_ids();
        assert(stops.size() == 3);
        assert(std::find(stops.begin(), stops.end(), 1u) != stops.end());
        assert(std::find(stops.begin(), stops.end(), 106u) != stops.end());
        assert(std::find(stops.begin(), stops.end(), 50u) != stops.end());
        std::cout << "  [PASS] Stop token set is {1, 106, 50}.\n";
    }

    // Test 3: Encoding without a vocabulary throws rather than inventing tokens.
    {
        bool threw = false;
        try {
            tok.encode("Hello", true);
        } catch (const gturbo::GTurboFormatError&) {
            threw = true;
        }
        assert(threw);
        std::cout << "  [PASS] encode() without a loaded vocabulary throws.\n";
    }

    // Test 3b: A missing vocabulary file throws rather than silently substituting one.
    {
        gturbo::Tokenizer missing_tok;
        bool threw = false;
        try {
            missing_tok.load_vocabulary("definitely_not_a_real_vocab_file.json");
        } catch (const gturbo::GTurboFormatError&) {
            threw = true;
        }
        assert(threw);
        assert(!missing_tok.is_loaded());
        std::cout << "  [PASS] load_vocabulary() on a missing file throws.\n";
    }

    // Test 4: BPE over a miniature tokenizer.json.
    // Vocabulary and merges are tiny but the structure is exactly the real file's, so this
    // exercises the scanner, the merge loop, byte fallback, and special-token splitting.
    const std::string mini = "mini_tokenizer.json";
    {
        std::ofstream out(mini, std::ios::binary);
        out << R"({
  "version": "1.0",
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<eos>", "special": true},
    {"id": 2, "content": "<bos>", "special": true},
    {"id": 3, "content": "<unk>", "special": true},
    {"id": 105, "content": "<|turn>", "special": true},
    {"id": 106, "content": "<turn|>", "special": true}
  ],
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
  "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}]},
  "model": {
    "type": "BPE",
    "byte_fallback": true,
    "vocab": {
      "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
      "<0x48>": 10, "<0x69>": 11, "<0xC3>": 12, "<0xA9>": 13,
      "H": 20, "i": 21, "Hi": 22,
      "▁": 30, "w": 31, "o": 32, "r": 33, "l": 34, "d": 35,
      "▁w": 36, "or": 37, "▁wor": 38, "ld": 39, "▁world": 40,
      "<|turn>": 105, "<turn|>": 106
    },
    "merges": [["H", "i"], ["▁", "w"], ["o", "r"], ["▁w", "or"],
               ["l", "d"], ["▁wor", "ld"]]
  }
})";
    }

    gturbo::Tokenizer bpe;
    assert(bpe.load_vocabulary(mini));
    assert(bpe.is_loaded());
    assert(bpe.bos_id() == 2 && bpe.eos_id() == 1);
    assert(bpe.end_of_turn_id() == 106);
    std::cout << "  [PASS] tokenizer.json scanner loaded vocab, merges and added tokens.\n";

    {
        // "Hi" merges H+i -> "Hi" (id 22), not two single-character tokens.
        auto t = bpe.encode("Hi", false);
        assert((t == std::vector<uint32_t>{22}));
        // add_bos prepends <bos>.
        auto tb = bpe.encode("Hi", true);
        assert((tb == std::vector<uint32_t>{2, 22}));
        std::cout << "  [PASS] BPE merges adjacent pairs by rank.\n";
    }

    {
        // Leading space becomes U+2581 and merges all the way to a single token.
        auto t = bpe.encode(" world", false);
        assert((t == std::vector<uint32_t>{40}));
        assert(bpe.decode(t, true) == " world");
        std::cout << "  [PASS] space normalization, full merge, and decode round-trip.\n";
    }

    {
        // Special tokens are matched verbatim and split the surrounding text.
        auto t = bpe.encode("<|turn>Hi<turn|>", false);
        assert((t == std::vector<uint32_t>{105, 22, 106}));
        // skip_special drops them; without it they are rendered.
        assert(bpe.decode(t, true) == "Hi");
        std::cout << "  [PASS] added tokens are matched verbatim and split the input.\n";
    }

    {
        // "e" is absent from this vocabulary, so byte fallback kicks in. U+00E9 is two
        // UTF-8 bytes and must be re-fused on decode, not emitted as two broken halves.
        auto t = bpe.encode("\xC3\xA9", false);
        assert((t == std::vector<uint32_t>{12, 13}));
        assert(bpe.decode(t, true) == "\xC3\xA9");
        std::cout << "  [PASS] byte fallback splits and re-fuses multi-byte UTF-8.\n";
    }

    {
        // The Gemma 4 chat template. These are NOT <start_of_turn>/<end_of_turn>.
        std::vector<gturbo::Tokenizer::ChatMessage> msgs{{"user", "Hi"}};
        std::string rendered = bpe.apply_chat_template(msgs);
        const std::string expected =
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        if (rendered != expected) {
            std::cout << "    got:      " << rendered << "\n";
            std::cout << "    expected: " << expected << "\n";
        }
        assert(rendered == expected);
        std::cout << "  [PASS] chat template renders the Gemma 4 turn markers.\n";
    }

    fs::remove(mini);

    // Test 5: Golden token IDs against the real tokenizer, when a bundle is present.
    // This single assertion catches almost every tokenizer regression. Pinned in the
    // reference at Tests/.../ChatTemplateTests.swift:85-87.
    const std::vector<std::string> candidates = {
        "gemma-4-26b-a4b.gturbo/tokenizer/tokenizer.json",
        "../gemma-4-26b-a4b.gturbo/tokenizer/tokenizer.json",
    };
    std::string real_path;
    for (const auto& c : candidates) {
        if (fs::exists(c)) { real_path = c; break; }
    }

    if (real_path.empty()) {
        std::cout << "  [SKIP] golden-token test: no .gturbo bundle with a tokenizer yet.\n";
    } else {
        gturbo::Tokenizer real;
        real.load_vocabulary(real_path);
        assert(real.vocab_size() == 262144);
        assert(real.bos_id() == 2);
        assert(real.eos_id() == 1);
        assert(real.end_of_turn_id() == 106);

        std::vector<gturbo::Tokenizer::ChatMessage> msgs{{"user", "Hi"}};
        auto ids = real.encode(real.apply_chat_template(msgs), false);
        const std::vector<uint32_t> golden = {
            2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107, 100, 45518, 107, 101};
        if (ids != golden) {
            std::cout << "    got:      ";
            for (auto t : ids) std::cout << t << " ";
            std::cout << "\n    expected: ";
            for (auto t : golden) std::cout << t << " ";
            std::cout << "\n";
        }
        assert(ids == golden);
        std::cout << "  [PASS] golden chat-template token IDs match the reference.\n";

        // Round-trip a sentence with punctuation and a non-ASCII character.
        const std::string sentence = "The quick brown fox jumps over 13 lazy dogs \xC3\xA9!";
        auto rt = real.encode(sentence, false);
        assert(real.decode(rt, true) == sentence);
        std::cout << "  [PASS] real-vocabulary encode/decode round-trip.\n";
    }

    std::cout << "[TEST SUCCESS] All Tokenizer unit tests PASSED!\n";
    return 0;
}
