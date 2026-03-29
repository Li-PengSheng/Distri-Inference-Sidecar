/* tokenizer.h – C ABI declarations for the rust_ops static library.
 *
 * Link with librust_ops.a (built via `cargo build --release` in rust_ops/).
 */

/* tokenize_len returns the number of whitespace-separated words in `input`,
 * used as a lightweight approximation of the token count.
 * Returns 0 if `input` is NULL. */
int tokenize_len(const char *input);

/* free_string releases a C string previously allocated by Rust inside
 * rust_ops. Do not call this on strings allocated elsewhere. */
void free_string(char *s);