# from llama_index import VectorStoreIndex, SimpleDirectoryReader
from typing import Dict
import nest_asyncio

nest_asyncio.apply()

from llama_index.core.extractors import BaseExtractor

from llama_index.core.node_parser import SimpleNodeParser  # type: ignore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader  # type: ignore
from llama_index.core import Settings  # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore

# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.extractors import (  # type: ignore
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
from constants import ALLOWED_VALUE, ENGINEERING_ROLE, FINANCE_ROLE
from llama_index.core.ingestion import IngestionPipeline  # type: ignore

from llama_index.llms.lmstudio import LMStudio  # type: ignore

Settings.llm = LMStudio(
    model_name="gemma-2-2b-instruct",
    base_url="http://localhost:1234/v1",
    temperature=0.3,
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Connect documents to their permissions based on directory
# In a real applications, this would come from the source api
# (eg, from GoogleDrive's file metadata)
documents = [
    ("engineering", [ENGINEERING_ROLE]),
    ("finance", [FINANCE_ROLE]),
    ("both", [ENGINEERING_ROLE, FINANCE_ROLE]),
]

nodes = []
for directory, roles in documents:

    class CustomExtractor(BaseExtractor):
        async def aextract(self, nodes) -> list[Dict]:
            metadata_list = [{role: ALLOWED_VALUE for role in roles}] * len(nodes)
            return metadata_list

    # Use the CustomExtractor to attach metadata to nodes based on their defined permissions
    extractor = [
        # TitleExtractor(nodes=5, llm=llm),
        # QuestionsAnsweredExtractor(questions=1, llm=llm),
        # EntityExtractor(prediction_threshold=0.5),
        # SummaryExtractor(summaries=["prev", "self"], llm=llm),
        # KeywordExtractor(keywords=10, llm=llm),
        CustomExtractor()
    ]

    transformations = extractor

    docs = SimpleDirectoryReader(f"documents/{directory}").load_data()
    pipeline = IngestionPipeline(transformations=transformations)

    uber_nodes = pipeline.run(documents=docs)
    print("\n================================")
    print(uber_nodes)
    print("\n================================")

    # parser = SimpleNodeParser.from_defaults(metadata_extractor=extractor)
    # nodes = nodes + parser.get_nodes_from_documents(docs)

# Create the index with all nodes including their role-based metadata
index = VectorStoreIndex(uber_nodes)

# Persist the index for querying in a different script to reduce OpenAI API usage
index.storage_context.persist()
print("Index persisted")
