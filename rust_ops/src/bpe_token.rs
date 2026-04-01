use std::collections::HashMap;

pub struct BPETokenizer {
    pub vocab: HashMap<String, u32>,
    pub merges: HashMap<(String, String), String>,
    pub merge_order: Vec<(String, String)>,
}

impl BPETokenizer {
    pub fn new(_vocab_size: usize) -> Self {
        BPETokenizer {
            vocab: HashMap::new(),
            merges: HashMap::new(),
            merge_order: Vec::new(),
        }
    }

    pub fn train(&mut self, text: &str, vocab_size: usize) {
        let mut words: Vec<Vec<String>> = text
            .split_whitespace()
            .map(|w| w.chars().map(|c| c.to_string()).collect())
            .collect();

        // 修复 vocab_id 借用问题：先收集所有字符再插入
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
            
            // 用 HashMap 存 merges，O(1) 查找
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

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();

        for word in text.split_whitespace() {
            let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // 按 merge_order 顺序，用 HashMap O(1) 查找
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