use std::collections::HashMap;

/// BPETokenizer implements a minimal corpus-trained Byte-Pair Encoding tokenizer.
///
/// It is intentionally a lightweight admission-control tokenizer, not a
/// model-compatible tokenizer: it starts from Unicode characters, splits input
/// on whitespace, and does not implement model-specific special tokens. Train
/// it on a representative corpus before using `encode` for token limits.
pub struct BPETokenizer {
    /// Maps base characters and learned merged strings to token IDs.
    pub vocab: HashMap<String, u32>,
    /// Maps a merge pair to its combined token for lookup during encoding.
    pub merges: HashMap<(String, String), String>,
    /// Preserves merge-learning order, which is semantically required when
    /// encoding because applying the same rules in another order can yield a
    /// different tokenization.
    pub merge_order: Vec<(String, String)>,
}

impl BPETokenizer {
    /// Creates an empty tokenizer.
    ///
    /// The `_vocab_size` parameter is retained for call-site symmetry with
    /// `train`; construction itself has no vocabulary-size side effect. Call
    /// `train` to populate the tokenizer.
    pub fn new(_vocab_size: usize) -> Self {
        BPETokenizer {
            vocab: HashMap::new(),
            merges: HashMap::new(),
            merge_order: Vec::new(),
        }
    }

    /// Learns character tokens and frequent-pair merges from `text` until the
    /// vocabulary reaches `vocab_size` or no adjacent pair remains.
    ///
    /// Training mutates `vocab`, `merges`, and `merge_order`; it does not clear
    /// existing state, so callers should train a new tokenizer rather than
    /// retraining one when they need an independent vocabulary. Equal-frequency
    /// pairs are selected through `HashMap` iteration, so their tie order is
    /// not a stable cross-run tokenizer contract.
    pub fn train(&mut self, text: &str, vocab_size: usize) {
        let mut words: Vec<Vec<String>> = text
            .split_whitespace()
            .map(|w| w.chars().map(|c| c.to_string()).collect())
            .collect();

        // Collect all unique characters first, then insert them into the vocab
        // in a single pass to avoid conflicting borrows on self.vocab.
        let mut all_chars: Vec<String> = Vec::new();
        for word in &words {
            for ch in word {
                if !self.vocab.contains_key(ch) {
                    all_chars.push(ch.clone());
                }
            }
        }
        for ch in all_chars {
            let id = self.vocab.len() as u32;
            self.vocab.entry(ch).or_insert(id);
        }

        while self.vocab.len() < vocab_size {
            let mut pair_freq: HashMap<(String, String), usize> = HashMap::new();
            for word in &words {
                for pair in word.windows(2) {
                    *pair_freq
                        .entry((pair[0].clone(), pair[1].clone()))
                        .or_insert(0) += 1;
                }
            }

            let best = match pair_freq.iter().max_by_key(|e| e.1) {
                Some((pair, _)) => pair.clone(),
                None => break,
            };

            let merged = format!("{}{}", best.0, best.1);

            // The map gives lookup by pair, while merge_order preserves the
            // learned sequence required to reproduce BPE merge semantics.
            self.merges.insert(best.clone(), merged.clone());
            self.merge_order.push(best.clone());

            let id = self.vocab.len() as u32;
            self.vocab.insert(merged.clone(), id);

            for word in &mut words {
                let mut i = 0;
                while i + 1 < word.len() {
                    if word[i] == best.0 && word[i + 1] == best.1 {
                        word[i] = merged.clone();
                        word.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }

    /// Encodes whitespace-delimited `text` into this tokenizer's token IDs.
    ///
    /// Merge rules are applied in training order because later rules may depend
    /// on tokens produced by earlier rules. Tokens absent from `vocab` become
    /// ID 0 rather than producing an error so admission can still count prompts
    /// containing unseen characters; callers needing model-tokenizer fidelity
    /// must use the model's tokenizer instead.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();

        for word in text.split_whitespace() {
            let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // Apply merges in training order; unordered map iteration would
            // change the result when one merge enables another.
            for (left, right) in &self.merge_order {
                let merged = match self.merges.get(&(left.clone(), right.clone())) {
                    Some(m) => m.clone(),
                    None => continue,
                };
                let mut i = 0;
                while i + 1 < tokens.len() {
                    if tokens[i] == *left && tokens[i + 1] == *right {
                        tokens[i] = merged.clone();
                        tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            for t in &tokens {
                result.push(*self.vocab.get(t).unwrap_or(&0));
            }
        }
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_builds_vocab() {
        let mut tok = BPETokenizer::new(20);
        tok.train("hello world hello foo bar", 20);
        assert!(!tok.vocab.is_empty());
    }

    #[test]
    fn test_encode_returns_ids() {
        let mut tok = BPETokenizer::new(20);
        tok.train("hello world hello foo bar", 20);
        let ids = tok.encode("hello world");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_encode_empty_string() {
        let mut tok = BPETokenizer::new(10);
        tok.train("hello world", 10);
        let ids = tok.encode("");
        assert_eq!(ids.len(), 0);
    }

    #[test]
    fn test_merge_reduces_tokens() {
        let mut tok = BPETokenizer::new(30);
        tok.train(&"hello world ".repeat(100), 30);
        let ids = tok.encode("hello");
        assert!(ids.len() <= 5);
    }
}
