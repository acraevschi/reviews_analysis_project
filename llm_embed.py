from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate, VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.json import JSONReader
from llama_index.core import Document
from datetime import datetime
import nest_asyncio
import os
import json

nest_asyncio.apply()

data_dir = "./data"

embed_model = HuggingFaceEmbedding("BAAI/bge-m3", trust_remote_code=True)
llm = Ollama(model="llama3.2")
Settings.llm = llm

reader = JSONReader(
    levels_back=None,
    collapse_length=False,
    ensure_ascii=False,
    is_jsonl=False,
    clean_json=True,
)

channels_folders = os.listdir(data_dir)

documents = []
for folder in channels_folders:
    channel_dir = os.path.join(data_dir, folder)
    files = os.listdir(channel_dir)
    with open(f"{channel_dir}/channel_metadata.json", "r", encoding="utf-8") as f:
        channel_metadata = json.load(f)

    channel_creator = channel_metadata.get("username", "")
    channel_creation_date = channel_metadata.get("creation_date", "")
    channel_viewcount = channel_metadata.get("view_count", "")
    channel_subscriber_count = channel_metadata.get("subscriber_count", "")
    channel_video_count = channel_metadata.get("video_count", "")
    for json_file in files:
        if json_file == "channel_metadata.json":
            continue

        file_path = os.path.join(channel_dir, json_file)

        with open(file_path, "r", encoding="utf-8") as f:
            video_info = json.load(f)
        video_metadata = video_info.get("video_metadata", {})
        video_date = video_metadata.get("published_at", "")
        video_viewcount = video_metadata.get("view_count", "")
        video_likecount = video_metadata.get("like_count", "")
        video_commentcount = video_metadata.get("comment_count", "")

        # Read JSON data into documents
        documents.append(
            Document(
                text=video_info.get("transcript", ""),
                metadata={
                    "channel_creator": channel_creator,
                    "channel_creation_date": channel_creation_date,
                    "channel_viewcount": channel_viewcount,
                    "channel_subscriber_count": channel_subscriber_count,
                    "channel_video_count": channel_video_count,
                    "video_publication_date": video_date,
                    "video_viewcount": video_viewcount,
                    "video_likecount": video_likecount,
                    "video_commentcount": int(video_commentcount),
                },
            )
        )
        # TODO: Figure out what to do with the comments. Mayeg use `relationships`

Settings.embed_model = embed_model

index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=5)

qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above I want you to think step by step to answer the query in a crisp manner, in case you don't know the answer, say 'I don't know!'.\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

response = query_engine.query(
    'When was the video published? The one that is titled "La falta de educación financiera - VisualEconomik".'
)

# TODO: How to use metadata more effectively?

print(response)
