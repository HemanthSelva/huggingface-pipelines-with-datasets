"""
# pip install transformers torch sentence-transformers datasets

Author: HEMANTHSELVA A K
Company: Sourcesys Technologies
Description: 8 HuggingFace Transformer pipeline tasks with datasets
"""

from transformers import pipeline
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

def print_divider(task_num, task_name):
    print(f"\n{'='*70}")
    print(f"Task {task_num}: {task_name}")
    print(f"{'='*70}")

def main():
    # 1. Question Answering
    print_divider(1, "Question Answering")
    qa_pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    # Handle potentially necessary `trust_remote_code` in newer datasets packages
    try:
        qa_data = load_dataset('squad', split='validation', trust_remote_code=True)
    except TypeError:
        qa_data = load_dataset('squad', split='validation')
        
    qa_sample = qa_data[0]
    print(f"[Dataset: squad] Context: {qa_sample['context'][:150]}...")
    print(f"[Dataset: squad] Question: {qa_sample['question']}")
    qa_res = qa_pipe(question=qa_sample['question'], context=qa_sample['context'])
    print(f"[Pipeline Output]: {qa_res}")
    
    
    # 2. Token Classification (NER)
    print_divider(2, "Token Classification (NER)")
    ner_pipe = pipeline("token-classification", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    try:
        ner_data = load_dataset('conll2003', split='test', trust_remote_code=True)
    except TypeError:
        ner_data = load_dataset('conll2003', split='test')
        
    ner_sample = ner_data[0]
    ner_text = " ".join(ner_sample['tokens'])
    print(f"[Dataset: conll2003] Text: {ner_text}")
    ner_res = ner_pipe(ner_text)
    print(f"[Pipeline Output] (First 3 entities): {ner_res[:3]}")


    # 3. Text Classification (Sentiment)
    print_divider(3, "Text Classification (Sentiment)")
    tc_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    try:
        tc_data = load_dataset('imdb', split='test', trust_remote_code=True)
    except TypeError:
        tc_data = load_dataset('imdb', split='test')
        
    tc_sample = tc_data[0]
    tc_text = tc_sample['text'][:400] + "..."
    print(f"[Dataset: imdb] Text: {tc_text}")
    tc_res = tc_pipe(tc_text)
    print(f"[Pipeline Output]: {tc_res}")


    # 4. Zero-Shot Classification
    print_divider(4, "Zero-Shot Classification")
    zsc_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    try:
        zsc_data = load_dataset('ag_news', split='test', trust_remote_code=True)
    except TypeError:
        zsc_data = load_dataset('ag_news', split='test')
        
    zsc_sample = zsc_data[0]
    zsc_text = zsc_sample['text']
    labels = ["world", "sports", "business", "science", "technology"]
    print(f"[Dataset: ag_news] Text: {zsc_text}")
    print(f"Candidate Labels: {labels}")
    zsc_res = zsc_pipe(zsc_text, candidate_labels=labels)
    print(f"[Pipeline Output] Top Label: {zsc_res['labels'][0]}, Score: {zsc_res['scores'][0]:.4f}")


    # 5. Summarization
    print_divider(5, "Summarization")
    sum_pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    try:
        sum_data = load_dataset('cnn_dailymail', '3.0.0', split='test', trust_remote_code=True)
    except TypeError:
        sum_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
        
    sum_sample = sum_data[0]
    sum_text = sum_sample['article'][:1000] # Truncation for efficiency
    print(f"[Dataset: cnn_dailymail v3.0.0] Article snippet: {sum_text[:150]}...")
    sum_res = sum_pipe(sum_text)
    print(f"[Pipeline Output]: {sum_res[0]['summary_text']}")


    # 6. Text Generation
    print_divider(6, "Text Generation")
    tg_pipe = pipeline("text-generation", model="gpt2")
    try:
        tg_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)
    except TypeError:
        tg_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        
    # Find a non-empty starting text length>50 for better seed structure
    tg_seed = ""
    for item in tg_data:
        if len(item['text'].strip()) > 50:
            tg_seed = item['text'].strip()[:100]
            break
    print(f"[Dataset: wikitext-2-raw-v1] Seed Text: {tg_seed}")
    tg_res = tg_pipe(tg_seed, max_new_tokens=30, num_return_sequences=1)
    print(f"[Pipeline Output]: {tg_res[0]['generated_text']}")


    # 7. Sentence Similarity
    print_divider(7, "Sentence Similarity")
    ss_model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        ss_data = load_dataset('sick', split='test', trust_remote_code=True)
    except TypeError:
        ss_data = load_dataset('sick', split='test')
        
    ss_sample = ss_data[0]
    sen_a = ss_sample['sentence_A']
    sen_b = ss_sample['sentence_B']
    print(f"[Dataset: sick] Sentence A: {sen_a}")
    print(f"[Dataset: sick] Sentence B: {sen_b}")
    emb1 = ss_model.encode(sen_a)
    emb2 = ss_model.encode(sen_b)
    cosine_sim = util.cos_sim(emb1, emb2)
    print(f"[Model Output] Cosine Similarity: {cosine_sim.item():.4f}")


    # 8. Feature Extraction
    print_divider(8, "Feature Extraction")
    fe_pipe = pipeline("feature-extraction", model="distilbert-base-uncased")
    try:
        # 'emotion' dataset sometimes needs trust_remote_code explicitly or redirects to 'dair-ai/emotion' 
        fe_data = load_dataset('emotion', split='test', trust_remote_code=True)
    except TypeError:
        fe_data = load_dataset('emotion', split='test')
    except Exception:
        try:
            fe_data = load_dataset('dair-ai/emotion', split='test', trust_remote_code=True)
        except TypeError:
            fe_data = load_dataset('dair-ai/emotion', split='test')
            
    fe_sample = fe_data[0]
    fe_text = fe_sample['text']
    print(f"[Dataset: emotion] Text: {fe_text}")
    fe_res = fe_pipe(fe_text)
    # Output is nested lists: [batch][sequence_length][hidden_dim]
    print(f"[Pipeline Output] Extracted feature tensor shape: Batch Size: {len(fe_res)}, Tokens: {len(fe_res[0])}, Hidden Dim: {len(fe_res[0][0])}")

if __name__ == "__main__":
    main()
