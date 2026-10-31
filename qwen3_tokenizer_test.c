/* Chat template truncation. "make tokenizertest" also checks that the one
 * cut below, and not the exact fit, printed the warning to stderr. */
#include "iris_qwen3.h"
#include <stdio.h>
#include <string.h>

int iris_verbose = 0;

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <tokenizer.json>\n", argv[0]);
        return 2;
    }
    qwen3_tokenizer_t *tok = qwen3_tokenizer_load(argv[1]);
    if (!tok) return 2;

    int full, fit, cut;
    int *whole = qwen3_tokenize_chat(tok, "a red apple", &full, 0, 0);
    int *exact = qwen3_tokenize_chat(tok, "a red apple", &fit, full, 0);
    int *head = qwen3_tokenize_chat(tok, "a red apple", &cut, full - 4, 0);

    if (fit != full || cut != full - 4 ||
        memcmp(whole, exact, fit * sizeof(int)) != 0 ||
        memcmp(whole, head, cut * sizeof(int)) != 0) {
        fprintf(stderr, "truncation mismatch: %d tokens, %d fitted, %d cut\n",
                full, fit, cut);
        return 1;
    }
    return 0;
}
