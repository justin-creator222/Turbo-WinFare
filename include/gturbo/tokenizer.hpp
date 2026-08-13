#ifndef GTURBO_TOKENIZER_HPP
#define GTURBO_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace gturbo {

enum class PieceType : uint8_t {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Byte = 5,
    Unused = 6
};

struct TokenPiece {
    std::string piece;
    float score{0.0f};
    PieceType type{PieceType::Normal};
    uint32_t id{0};
};

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Loads a HuggingFace tokenizer.json (BPE with byte fallback). Throws on any failure --
    // there is no synthesized-vocabulary fallback.
    bool load_vocabulary(const std::string& vocab_file = "");

    std::vector<uint32_t> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<uint32_t>& tokens, bool skip_special = true) const;
    // NOTE: not byte-fallback-safe on its own -- a lone <0xNN> decodes to its literal piece
    // text, and a multi-byte character split across several of them will not reassemble.
    // For token-by-token output use IncrementalDetokenizer (gturbo/detokenizer.hpp).
    std::string decode_single(uint32_t token, bool skip_special = true) const;

    // True when `token` is a <0xNN> byte-fallback piece, with its byte in `value`.
    bool is_byte_fallback(uint32_t token, uint8_t& value) const;
    bool is_special_token(uint32_t token) const {
        return token < is_special_.size() && is_special_[token];
    }

    struct ChatMessage {
        std::string role;     // "user", "model", "system"
        std::string content;
    };

    // Renders the Gemma 4 chat template. For a single user turn "Hi" this produces
    //   <bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
    // and encodes (with add_bos=false, since the template already carries <bos>) to
    //   [2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107, 100, 45518, 107, 101]
    //
    // Note these are NOT the Gemma 2/3 <start_of_turn>/<end_of_turn> markers; the trailing
    // <|channel>thought<channel|> is the "thinking disabled" generation prompt.
    std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;

    uint32_t bos_id() const { return bos_id_; }
    uint32_t eos_id() const { return eos_id_; }
    uint32_t pad_id() const { return pad_id_; }
    uint32_t unk_id() const { return unk_id_; }
    uint32_t end_of_turn_id() const { return end_of_turn_id_; }
    uint32_t vocab_size() const { return vocab_size_; }
    bool is_loaded() const { return loaded_; }

    // Generation halts on any of these; the stop token itself is never emitted.
    const std::vector<uint32_t>& stop_token_ids() const { return stop_token_ids_; }

private:
    // Gemma 4 special-token IDs, per the Swift reference
    // (Sources/TurboFieldfare/Tokenization/Tokenizer.swift:101-135 in drumih/turbo-fieldfare).
    // Note these are NOT the Gemma 2/3 <start_of_turn>/<end_of_turn> markers.
    uint32_t bos_id_{2};             // <bos>
    uint32_t eos_id_{1};             // <eos>
    uint32_t pad_id_{0};             // <pad>
    uint32_t unk_id_{3};             // <unk>
    uint32_t turn_start_id_{105};    // <|turn>
    uint32_t end_of_turn_id_{106};   // <turn|>
    uint32_t tool_response_id_{50};  // <|tool_response>
    uint32_t vocab_size_{262144};
    bool loaded_{false};

    std::vector<uint32_t> stop_token_ids_{1, 106, 50};

    // id -> piece text, and the reverse map.
    std::vector<std::string> pieces_;
    std::unordered_map<std::string, uint32_t> piece_map_;
    // Byte-fallback tokens: 0x41 -> id of "<0x41>".
    std::unordered_map<uint8_t, uint32_t> byte_to_id_;
    // BPE merge priorities: "left\x00right" -> rank (lower merges first).
    std::unordered_map<std::string, uint32_t> merge_ranks_;
    // Added/special tokens, matched verbatim before BPE runs. Sorted longest-first so
    // "<|tool_response>" wins over any shorter prefix.
    std::vector<std::pair<std::string, uint32_t>> added_tokens_;
    std::vector<bool> is_special_;

    bool parse_tokenizer_json(const std::string& filepath);
    // Runs BPE over one segment of already-normalized text (spaces already U+2581).
    void bpe_segment(const std::string& text, std::vector<uint32_t>& out) const;
    void emit_byte_fallback(const std::string& piece, std::vector<uint32_t>& out) const;
};

} // namespace gturbo

#endif // GTURBO_TOKENIZER_HPP
