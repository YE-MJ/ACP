from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import os 

# ProtBERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')
model = BertModel.from_pretrained('Rostlab/prot_bert')

# 텍스트 파일 읽기 함수
def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        current_label = None
        current_sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_label:
                    sequences[current_label] = ''.join(current_sequence)
                current_label = line[1:]  # '>' 제거
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_label:
            sequences[current_label] = ''.join(current_sequence)
    return sequences

# 단백질 서열 파일 경로

file_path = './independent_test_data/independent_ACP_data.txt'

# 파일에서 단백질 서열 읽기
sequences = read_fasta(file_path)

# 단백질 서열 임베딩
embeddings = []

for label, sequence in sequences.items():
    # ProtBERT는 아미노산 단위로 토큰화해야 합니다.
    sequence = ' '.join(sequence)

    # 토크나이징
    inputs = tokenizer(sequence, return_tensors='pt')

    # 모델에 입력하여 임베딩 추출
    with torch.no_grad():
        outputs = model(**inputs)

    # 임베딩 벡터 추출
    embedding = outputs.last_hidden_state

    # 시퀀스 레벨 임베딩을 위해 평균을 취합니다.
    sequence_embedding = torch.mean(embedding, dim=1).squeeze().numpy()

    # 라벨과 임베딩을 리스트에 저장
    embeddings.append([label] + sequence_embedding.tolist())

# 결과를 DataFrame으로 변환
df = pd.DataFrame(embeddings)

# 첫 번째 열은 라벨로 설정
df.columns = ['name'] + [f'embedding_{i}' for i in range(1, df.shape[1])]
df['target'] = df['name'].apply(lambda x: 1 if 'positive' in x else 0)


csv_path = './ACP_feature/embedding/'

# 디렉토리가 없으면 생성
if not os.path.exists(csv_path):
    os.makedirs(csv_path)


# CSV 파일로 저장
csv_file_path = './feature/embedding/independent/protein_embeddings_protbert.csv'

df.to_csv(csv_file_path, index=False)

print(f"Embedding results saved to {csv_file_path}")