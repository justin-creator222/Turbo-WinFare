#ifndef GTURBO_C_API_H
#define GTURBO_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef TURBO_BUILD_DLL
    #define TURBO_API __declspec(dllexport)
  #else
    #define TURBO_API __declspec(dllimport)
  #endif
#else
  #define TURBO_API __attribute__((visibility("default")))
#endif

/* Returns NULL on failure; call turbo_engine_last_error() for the reason. */
TURBO_API void* turbo_engine_create(const char* model_dir);
TURBO_API void turbo_engine_destroy(void* handle);

/*
 * Returns generated text, or NULL on failure -- callers must check. Error detail goes to
 * turbo_engine_last_error(), never into the returned string: a caller that renders the
 * return value as an assistant reply must not be able to display an error as model output.
 */
TURBO_API const char* turbo_engine_generate(void* handle, const char* prompt, int max_tokens);
TURBO_API const char* turbo_engine_get_telemetry(void* handle);

/*
 * Per-request generation options. Zero-initializing this struct is NOT valid -- a zero top_k
 * with top_p < 1 is rejected. Call turbo_options_default() first, then override.
 */
typedef struct {
    int max_tokens;
    float temperature;          /* 0 = greedy */
    float top_p;
    int top_k;
    float repetition_penalty;
    int has_seed;               /* non-zero to use `seed` */
    unsigned long long seed;
    const char* system_prompt;  /* may be NULL */
    const char* const* stop_strings;
    int stop_count;
} TurboGenerationOptions;

TURBO_API void turbo_options_default(TurboGenerationOptions* out);

/*
 * Called once per generated token with the newly visible text. Return 0 to cancel; the
 * generation then ends with stop reason "cancelled" rather than an error.
 */
typedef int (*TurboTokenCallback)(void* user, int index, unsigned int token_id, const char* delta);

/*
 * Full-featured generation. `messages_json` is an OpenAI-shaped array,
 *   [{"role":"user","content":"..."}, ...]
 * which keeps this ABI stable as new roles appear. Passing a struct array would have frozen
 * the message shape into the ABI.
 *
 * `opts` may be NULL for defaults; `on_token` may be NULL for non-streaming use.
 * Returns the complete text, or NULL on failure (see turbo_engine_last_error()).
 */
TURBO_API const char* turbo_engine_generate_ex(void* handle, const char* messages_json,
                                               const TurboGenerationOptions* opts,
                                               TurboTokenCallback on_token, void* user);

/* Why the last generation ended: 0 eos, 1 end_of_turn, 2 stop_string, 3 max_tokens, 4 cancelled. */
TURBO_API int turbo_engine_last_stop_reason(void* handle);

/* "lfu" or "lru"; anything else is ignored. */
TURBO_API void turbo_engine_set_eviction_policy(void* handle, const char* policy);

/*
 * Last error message for this thread, or "" if the last call succeeded. The pointer stays
 * valid until the next failing call on the same thread.
 */
TURBO_API const char* turbo_engine_last_error(void);

TURBO_API int turbo_engine_load_model(void* handle, const char* model_dir);
TURBO_API void turbo_engine_unload_model(void* handle);
TURBO_API void turbo_engine_clear_cache(void* handle);
TURBO_API void turbo_engine_stop(void* handle);

#ifdef __cplusplus
}
#endif

#endif // GTURBO_C_API_H
