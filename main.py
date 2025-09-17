import streamlit as st
import pymongo
from pymongo import MongoClient
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from typing import List, Dict, Tuple
import io
import hashlib
import re
from datetime import datetime
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILE = "pdf_chat_config.json"

class PDFVectorChat:
    def __init__(self):
        self.model = None
        self.client = None
        self.db = None
        self.collection = None
        self.groups_collection = None
        self.openai_client = None
        self._model_loaded = False
        
    def load_config(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
        return {}
    
    def save_config(self, config):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
    
    def initialize_model_once(self):
        if not self._model_loaded:
            try:
                with st.spinner("Loading embedding model (one-time setup)..."):
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    self._model_loaded = True
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                return False
        return True
        
    def initialize_components(self, mongo_uri: str, db_name: str, collection_name: str, openai_api_key: str = ""):
        try:
            if not self.initialize_model_once():
                return False
            
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.groups_collection = self.db[f"{collection_name}_groups"]
            
            self.client.admin.command('ping')
            
            self._ensure_vector_index()
            
            if openai_api_key and openai_api_key.strip():
                openai.api_key = openai_api_key
                self.openai_client = openai
            
            config = {
                "mongo_uri": mongo_uri,
                "db_name": db_name,
                "collection_name": collection_name,
                "openai_api_key": openai_api_key,
                "initialized": True
            }
            self.save_config(config)
            
            return True
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return False
    
    def auto_initialize_from_config(self):
        config = self.load_config()
        if config.get("initialized", False):
            try:
                return self.initialize_components(
                    config.get("mongo_uri", ""),
                    config.get("db_name", ""),
                    config.get("collection_name", ""),
                    config.get("openai_api_key", "")
                )
            except Exception as e:
                logger.error(f"Auto-initialization failed: {str(e)}")
                return False
        return False
    
    def _ensure_vector_index(self):
        try:
            indexes = list(self.collection.list_indexes())
            vector_index_exists = any(index.get('name') == 'vector_index' for index in indexes)
            
            if not vector_index_exists:
                self.collection.create_index([("embedding", "2dsphere")], name="vector_index")
                logger.info("Created vector search index")
        except Exception as e:
            logger.warning(f"Could not create vector index: {str(e)}")
    
    def get_existing_pdfs(self) -> List[str]:
        try:
            existing_files = self.collection.distinct("filename")
            return sorted(existing_files)
        except Exception as e:
            logger.error(f"Error getting existing PDFs: {str(e)}")
            return []
    
    def create_group(self, group_name: str, pdf_files: List[str], description: str = "") -> bool:
        try:
            group_doc = {
                "group_name": group_name,
                "pdf_files": pdf_files,
                "description": description,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            existing_group = self.groups_collection.find_one({"group_name": group_name})
            if existing_group:
                st.error(f"Group '{group_name}' already exists!")
                return False
            
            self.groups_collection.insert_one(group_doc)
            st.success(f"Group '{group_name}' created successfully!")
            return True
        except Exception as e:
            st.error(f"Error creating group: {str(e)}")
            return False
    
    def get_groups(self) -> List[Dict]:
        try:
            groups = list(self.groups_collection.find({}, {"group_name": 1, "pdf_files": 1, "description": 1, "created_at": 1}))
            return groups
        except Exception as e:
            logger.error(f"Error getting groups: {str(e)}")
            return []
    
    def update_group(self, group_name: str, pdf_files: List[str], description: str = "") -> bool:
        try:
            self.groups_collection.update_one(
                {"group_name": group_name},
                {
                    "$set": {
                        "pdf_files": pdf_files,
                        "description": description,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            st.success(f"Group '{group_name}' updated successfully!")
            return True
        except Exception as e:
            st.error(f"Error updating group: {str(e)}")
            return False
    
    def delete_group(self, group_name: str) -> bool:
        try:
            result = self.groups_collection.delete_one({"group_name": group_name})
            if result.deleted_count > 0:
                st.success(f"Group '{group_name}' deleted successfully!")
                return True
            else:
                st.error(f"Group '{group_name}' not found!")
                return False
        except Exception as e:
            st.error(f"Error deleting group: {str(e)}")
            return False
    
    def get_group_by_name(self, group_name: str) -> Dict:
        try:
            group = self.groups_collection.find_one({"group_name": group_name})
            return group if group else {}
        except Exception as e:
            logger.error(f"Error getting group: {str(e)}")
            return {}
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        if not text:
            return []
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            if end < text_length:
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    word_end = text.rfind(' ', start, end)
                    if word_end != -1 and word_end > start + chunk_size // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        try:
            if not self._model_loaded:
                self.initialize_model_once()
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []
    
    def store_vectors_in_mongodb(self, chunks: List[str], filename: str) -> bool:
        try:
            file_hash = hashlib.md5(str(chunks).encode()).hexdigest()
            
            existing = self.collection.find_one({"file_hash": file_hash})
            if existing:
                st.warning("This PDF has already been processed.")
                return True
            
            documents = []
            progress_bar = st.progress(0, text="Processing chunks...")
            for i, chunk in enumerate(chunks):
                embedding = self.generate_embedding(chunk)
                if embedding:
                    doc = {
                        "filename": filename,
                        "chunk_index": i,
                        "text": chunk,
                        "embedding": embedding,
                        "file_hash": file_hash,
                        "created_at": datetime.utcnow()
                    }
                    documents.append(doc)
                
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress, text=f"Processing chunk {i+1}/{len(chunks)}")
            
            if documents:
                self.collection.insert_many(documents)
                progress_bar.progress(1.0, text="Complete!")
                st.success(f"Successfully stored {len(documents)} chunks in MongoDB")
                return True
            else:
                st.error("No valid embeddings generated")
                return False
                
        except Exception as e:
            st.error(f"Error storing vectors in MongoDB: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, filename: str = None, pdf_files: List[str] = None, top_k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            filter_criteria = {}
            if filename:
                filter_criteria["filename"] = filename
            elif pdf_files:
                filter_criteria["filename"] = {"$in": pdf_files}
            
            all_docs = list(self.collection.find(filter_criteria, {"text": 1, "embedding": 1, "filename": 1, "chunk_index": 1}))
            
            if not all_docs:
                return []
            
            similarities = []
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            for doc in all_docs:
                doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
                similarities.append({
                    "text": doc["text"],
                    "similarity": float(similarity),
                    "filename": doc["filename"],
                    "chunk_index": doc["chunk_index"],
                    "_id": doc["_id"]
                })
            
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def rank_with_llm(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if not self.openai_client or not chunks:
            return chunks
        
        try:
            context_text = "\n\n".join([f"Chunk {i+1} (from {chunk['filename']}): {chunk['text']}" for i, chunk in enumerate(chunks)])
            
            prompt = f"""
            Given the following query and text chunks from multiple PDFs, rank the chunks from most relevant to least relevant for answering the query.
            Consider semantic meaning, context, and how well each chunk addresses the query.
            
            Query: "{query}"
            
            Text Chunks:
            {context_text}
            
            Please provide your ranking as a comma-separated list of chunk numbers (e.g., "3,1,5,2,4") and briefly explain your reasoning.
            """
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that ranks text chunks by relevance to queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            llm_response = response.choices[0].message.content
            
            ranking_line = [line for line in llm_response.split('\n') if any(char.isdigit() for char in line)][0]
            ranking = [int(x.strip()) - 1 for x in ranking_line.split(',') if x.strip().isdigit()]
            
            if len(ranking) == len(chunks) and all(0 <= r < len(chunks) for r in ranking):
                reordered_chunks = [chunks[i] for i in ranking]
                for i, chunk in enumerate(reordered_chunks):
                    chunk['llm_rank'] = i + 1
                return reordered_chunks
            
        except Exception as e:
            logger.error(f"Error ranking with LLM: {str(e)}")
        
        return chunks
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        if not self.openai_client or not relevant_chunks:
            result = ""
            sources_seen = set()
            for chunk in relevant_chunks:
                if chunk["filename"] not in sources_seen:
                    result += f"\n\n**From {chunk['filename']}:**\n"
                    sources_seen.add(chunk["filename"])
                result += chunk["text"] + "\n"
            return result
        
        try:
            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                context_parts.append(f"[Source {i+1}: {chunk['filename']}]\n{chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""
            Based on the following context from multiple PDF documents, please answer the user's question.
            Each piece of context is labeled with its source PDF file.
            
            When providing your answer:
            1. Synthesize information from the relevant sources
            2. Include citations by mentioning the source PDF filename in your response
            3. If information comes from multiple sources, mention all relevant sources
            4. If the context doesn't contain enough information to answer the question, please say so
            
            Context:
            {context}
            
            Question: {query}
            
            Answer (with source citations):
            """
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context and always includes source citations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            sources = list(set([chunk["filename"] for chunk in relevant_chunks]))
            source_summary = f"\n\n**Sources referenced:** {', '.join(sources)}"
            
            return answer + source_summary
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Sorry, I couldn't generate an answer at this time."

@st.cache_resource
def get_pdf_chat_instance():
    return PDFVectorChat()

def initialize_session_state():
    if 'pdf_chat' not in st.session_state:
        st.session_state.pdf_chat = get_pdf_chat_instance()
    
    config = st.session_state.pdf_chat.load_config()
    
    if 'mongo_uri' not in st.session_state:
        st.session_state.mongo_uri = config.get("mongo_uri", "mongodb://localhost:27017/")
    if 'db_name' not in st.session_state:
        st.session_state.db_name = config.get("db_name", "pdf_chat_db")
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = config.get("collection_name", "pdf_vectors")
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = config.get("openai_api_key", "")
    
    if 'initialized' not in st.session_state:
        if config.get("initialized", False):
            with st.spinner("Restoring previous session..."):
                st.session_state.initialized = st.session_state.pdf_chat.auto_initialize_from_config()
        else:
            st.session_state.initialized = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_page' not in st.session_state:
        if st.session_state.initialized:
            st.session_state.current_page = 'main_menu'
        else:
            st.session_state.current_page = 'setup'
    if 'selected_pdf' not in st.session_state:
        st.session_state.selected_pdf = None
    if 'selected_group' not in st.session_state:
        st.session_state.selected_group = None
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = 'single'

def setup_page():
    st.title(" PDF Chat with MongoDB Vector Database")
    st.markdown("Configure your system and start chatting with your documents!")
    
    if st.session_state.initialized:
        st.success(" System is already initialized!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Go to Main Menu", type="primary"):
                st.session_state.current_page = 'main_menu'
                st.rerun()
        with col2:
            if st.button(" Reconfigure"):
                st.session_state.initialized = False
                st.rerun()
        return
    
    st.header(" System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mongo_uri = st.text_input(
            "MongoDB URI",
            value=st.session_state.mongo_uri,
            help="Your MongoDB connection string"
        )
        
        db_name = st.text_input(
            "Database Name",
            value=st.session_state.db_name,
            help="Name of the MongoDB database"
        )
    
    with col2:
        collection_name = st.text_input(
            "Collection Name",
            value=st.session_state.collection_name,
            help="Name of the collection to store vectors"
        )
        
        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)",
            value=st.session_state.openai_api_key,
            type="password",
            help="For LLM ranking and answer generation. Leave empty to use basic functionality."
        )
    
    if st.button(" Initialize System", type="primary", use_container_width=True):
        st.session_state.mongo_uri = mongo_uri
        st.session_state.db_name = db_name
        st.session_state.collection_name = collection_name
        st.session_state.openai_api_key = openai_api_key
        
        success = st.session_state.pdf_chat.initialize_components(
            mongo_uri, db_name, collection_name, openai_api_key
        )
        st.session_state.initialized = success
        
        if success:
            st.success(" System initialized successfully!")
            st.session_state.current_page = 'main_menu'
            st.rerun()
        else:
            st.error(" Failed to initialize system")

def main_menu_page():
    st.title(" PDF Chat System - Main Menu")
    
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button(" Settings"):
            st.session_state.current_page = 'setup'
            st.rerun()
    
    st.markdown("### Choose your chat mode:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####  Single PDF Chat")
        st.markdown("Chat with individual PDF documents")
        if st.button(" Single PDF Chat", type="primary", use_container_width=True):
            st.session_state.chat_mode = 'single'
            st.session_state.current_page = 'pdf_selection'
            st.rerun()
    
    with col2:
        st.markdown("####  Group PDF Chat")
        st.markdown("Create groups and chat with multiple PDFs")
        if st.button(" Group PDF Chat", type="primary", use_container_width=True):
            st.session_state.chat_mode = 'group'
            st.session_state.current_page = 'group_management'
            st.rerun()
    
    st.markdown("---")
    st.markdown("### System Statistics")
    
    try:
        total_pdfs = len(st.session_state.pdf_chat.get_existing_pdfs())
        total_groups = len(st.session_state.pdf_chat.get_groups())
        total_chunks = st.session_state.pdf_chat.collection.count_documents({})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total PDFs", total_pdfs)
        with col2:
            st.metric("Total Groups", total_groups)
        with col3:
            st.metric("Total Chunks", total_chunks)
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

def group_management_page():
    st.title(" PDF Group Management")
    
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button(" Main Menu"):
            st.session_state.current_page = 'main_menu'
            st.rerun()
    
    tab1, tab2, tab3 = st.tabs([" Existing Groups", " Create Group", "Upload PDFs"])
    
    with tab1:
        existing_groups_tab()
    
    with tab2:
        create_group_tab()
    
    with tab3:
        upload_pdfs_tab()

def existing_groups_tab():
    groups = st.session_state.pdf_chat.get_groups()
    
    if groups:
        st.markdown("### Choose a group to chat with:")
        
        for group in groups:
            with st.expander(f"📁 {group['group_name']} ({len(group['pdf_files'])} PDFs)"):
                if group.get('description'):
                    st.markdown(f"**Description:** {group['description']}")
                
                st.markdown("**PDFs in this group:**")
                for pdf in group['pdf_files']:
                    st.markdown(f"• {pdf}")
                
                st.markdown(f"**Created:** {group['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("💬 Chat", key=f"chat_group_{group['group_name']}"):
                        st.session_state.selected_group = group['group_name']
                        st.session_state.current_page = 'group_chat'
                        st.session_state.chat_history = []
                        st.rerun()
                
                with col2:
                    if st.button("✏️ Edit", key=f"edit_group_{group['group_name']}"):
                        st.session_state.editing_group = group['group_name']
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_group_{group['group_name']}"):
                        if st.session_state.pdf_chat.delete_group(group['group_name']):
                            st.rerun()
                
                if hasattr(st.session_state, 'editing_group') and st.session_state.editing_group == group['group_name']:
                    st.markdown("---")
                    st.markdown("**Edit Group:**")
                    
                    available_pdfs = st.session_state.pdf_chat.get_existing_pdfs()
                    
                    new_description = st.text_area(
                        "Description", 
                        value=group.get('description', ''),
                        key=f"edit_desc_{group['group_name']}"
                    )
                    
                    new_selected_pdfs = st.multiselect(
                        "Select PDFs for this group",
                        available_pdfs,
                        default=group['pdf_files'],
                        key=f"edit_pdfs_{group['group_name']}"
                    )
                    
                    col_edit1, col_edit2 = st.columns(2)
                    with col_edit1:
                        if st.button("💾 Save Changes", key=f"save_{group['group_name']}"):
                            if st.session_state.pdf_chat.update_group(group['group_name'], new_selected_pdfs, new_description):
                                if hasattr(st.session_state, 'editing_group'):
                                    delattr(st.session_state, 'editing_group')
                                st.rerun()
                    
                    with col_edit2:
                        if st.button("❌ Cancel", key=f"cancel_{group['group_name']}"):
                            if hasattr(st.session_state, 'editing_group'):
                                delattr(st.session_state, 'editing_group')
                            st.rerun()
    else:
        st.info("No groups found. Create your first group to get started!")

def create_group_tab():
    st.markdown("### Create a new PDF group:")
    
    available_pdfs = st.session_state.pdf_chat.get_existing_pdfs()
    
    if not available_pdfs:
        st.warning("No PDFs available. Please upload some PDFs first.")
        return
    
    group_name = st.text_input("Group Name", placeholder="e.g., Research Papers, Legal Documents, etc.")
    description = st.text_area("Description (Optional)", placeholder="Brief description of this group...")
    
    selected_pdfs = st.multiselect(
        "Select PDFs for this group",
        available_pdfs,
        help="Choose multiple PDFs to include in this group"
    )
    
    if st.button(" Create Group", type="primary", disabled=not (group_name and selected_pdfs)):
        if st.session_state.pdf_chat.create_group(group_name, selected_pdfs, description):
            st.rerun()

def upload_pdfs_tab():
    st.markdown("### Upload new PDFs:")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple PDF documents"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) uploaded")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        with col2:
            overlap = st.slider("Overlap", 50, 300, 100, 50)
        
        if st.button("🔄 Process All PDFs", type="primary"):
            with st.spinner("Processing multiple PDFs..."):
                success_count = 0
                for uploaded_file in uploaded_files:
                    st.info(f"Processing {uploaded_file.name}...")
                    
                    text = st.session_state.pdf_chat.extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        chunks = st.session_state.pdf_chat.chunk_text(text, chunk_size, overlap)
                        
                        success = st.session_state.pdf_chat.store_vectors_in_mongodb(chunks, uploaded_file.name)
                        
                        if success:
                            success_count += 1
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                    else:
                        st.error(f"❌ Could not extract text from {uploaded_file.name}")
                
                if success_count > 0:
                    st.success(f"✅ Successfully processed {success_count}/{len(uploaded_files)} PDFs!")
                    if success_count == len(uploaded_files):
                        st.balloons()

def pdf_selection_page():
    st.title(" Select or Upload PDF")
    
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("Main Menu"):
            st.session_state.current_page = 'main_menu'
            st.rerun()
    
    existing_pdfs = st.session_state.pdf_chat.get_existing_pdfs()
    
    tab1, tab2 = st.tabs([" Existing PDFs", " Upload New PDF"])
    
    with tab1:
        if existing_pdfs:
            st.markdown("### Choose from existing PDFs:")
            
            for pdf in existing_pdfs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f" **{pdf}**")
                with col2:
                    if st.button(" Chat", key=f"chat_{pdf}"):
                        st.session_state.selected_pdf = pdf
                        st.session_state.current_page = 'single_chat'
                        st.session_state.chat_history = []
                        st.rerun()
        else:
            st.info("No PDFs found in the database. Please upload a new PDF.")
    
    with tab2:
        st.markdown("### Upload a new PDF:")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            st.success(f" File uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
            with col2:
                overlap = st.slider("Overlap", 50, 300, 100, 50)
            
            if st.button(" Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    text = st.session_state.pdf_chat.extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        st.info(f" Extracted {len(text)} characters")
                        
                        chunks = st.session_state.pdf_chat.chunk_text(text, chunk_size, overlap)
                        st.info(f" Created {len(chunks)} chunks")
                        
                        success = st.session_state.pdf_chat.store_vectors_in_mongodb(chunks, uploaded_file.name)
                        
                        if success:
                            st.success(" PDF processed and stored successfully!")
                            st.session_state.selected_pdf = uploaded_file.name
                            if st.button(" Start Chatting"):
                                st.session_state.current_page = 'single_chat'
                                st.session_state.chat_history = []
                                st.rerun()
                        else:
                            st.error(" Failed to process PDF")
                    else:
                        st.error(" Could not extract text from PDF")

def single_chat_page():
    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
    with col_nav1:
        if st.button(" PDFs"):
            st.session_state.current_page = 'pdf_selection'
            st.rerun()
    with col_nav2:
        st.title(f" Chat with: {st.session_state.selected_pdf}")
    with col_nav3:
        if st.button(" Clear"):
            st.session_state.chat_history = []
            st.rerun()
    
    chat_interface(is_group_chat=False)

def group_chat_page():
    group = st.session_state.pdf_chat.get_group_by_name(st.session_state.selected_group)
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
    with col_nav1:
        if st.button(" Groups"):
            st.session_state.current_page = 'group_management'
            st.rerun()
    with col_nav2:
        st.title(f" Group Chat: {st.session_state.selected_group}")
    with col_nav3:
        if st.button(" Clear"):
            st.session_state.chat_history = []
            st.rerun()
    
    if group:
        with st.expander(" Group Information", expanded=False):
            if group.get('description'):
                st.markdown(f"**Description:** {group['description']}")
            st.markdown(f"**PDFs in this group ({len(group['pdf_files'])}):**")
            for pdf in group['pdf_files']:
                st.markdown(f"• {pdf}")
    
    chat_interface(is_group_chat=True)

def chat_interface(is_group_chat=False):
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        
        for i, (user_msg, agent_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong> User:</strong> {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: #e8f4fd; padding: 10px; border-radius: 10px; margin: 5px 0 20px 0;">
                <strong> Agent:</strong> {agent_msg}
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
    
    with st.sidebar:
        st.header("Chat Settings")
        top_k = st.slider("Top K Results", 1, 20, 10)
        use_llm_ranking = st.checkbox("Use LLM Ranking", value=bool(st.session_state.pdf_chat.openai_client))
        
        if is_group_chat:
            st.checkbox("Show Source Citations", value=True, disabled=True, help="Always enabled for group chat")
        else:
            show_citations = st.checkbox("Show Source Citations", value=False)
        
        st.markdown("---")
        if is_group_chat:
            group = st.session_state.pdf_chat.get_group_by_name(st.session_state.selected_group)
            st.markdown("**Current Group:**")
            st.info(st.session_state.selected_group)
            if group:
                st.markdown(f"**PDFs ({len(group['pdf_files'])}):**")
                for pdf in group['pdf_files']:
                    st.markdown(f"• {pdf}")
        else:
            st.markdown("**Current PDF:**")
            st.info(st.session_state.selected_pdf)
        
        st.markdown("---")
        st.markdown("**Configuration Status:**")
        config = st.session_state.pdf_chat.load_config()
        if config.get("mongo_uri"):
            st.success(" MongoDB Connected")
        else:
            st.error(" MongoDB Not Connected")
            
        if config.get("openai_api_key"):
            st.success(" OpenAI Configured")
        else:
            st.warning(" OpenAI Not Configured")
        
        if st.session_state.initialized:
            try:
                if is_group_chat:
                    group = st.session_state.pdf_chat.get_group_by_name(st.session_state.selected_group)
                    if group:
                        total_chunks = st.session_state.pdf_chat.collection.count_documents(
                            {"filename": {"$in": group['pdf_files']}}
                        )
                        st.metric("Total Chunks in Group", total_chunks)
                else:
                    total_chunks = st.session_state.pdf_chat.collection.count_documents(
                        {"filename": st.session_state.selected_pdf}
                    )
                    st.metric("Chunks in this PDF", total_chunks)
            except:
                pass
    
    st.markdown("### Ask a Question")
    
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input(
            "Type your question here:",
            placeholder="What is the main topic discussed in this document?" if not is_group_chat else "What are the common themes across these documents?",
            key="chat_input"
        )
        
        submit_button = st.form_submit_button(" Ask", type="primary", use_container_width=True)
        
        if submit_button and query.strip():
            with st.spinner("🔍 Searching and generating answer..."):
                if is_group_chat:
                    group = st.session_state.pdf_chat.get_group_by_name(st.session_state.selected_group)
                    if group:
                        similar_chunks = st.session_state.pdf_chat.search_similar_chunks(
                            query, 
                            pdf_files=group['pdf_files'], 
                            top_k=top_k
                        )
                    else:
                        similar_chunks = []
                else:
                    similar_chunks = st.session_state.pdf_chat.search_similar_chunks(
                        query, 
                        st.session_state.selected_pdf, 
                        top_k=top_k
                    )
                
                if similar_chunks:
                    if use_llm_ranking and st.session_state.pdf_chat.openai_client:
                        similar_chunks = st.session_state.pdf_chat.rank_with_llm(query, similar_chunks)
                    
                    answer = st.session_state.pdf_chat.generate_answer(query, similar_chunks)
                    
                    if not is_group_chat and show_citations:
                        sources = list(set([chunk["filename"] for chunk in similar_chunks]))
                        answer += f"\n\n**Source:** {', '.join(sources)}"
                    
                    st.session_state.chat_history.append((query.strip(), answer))
                    
                    st.rerun()
                else:
                    error_msg = "No relevant content found in the selected documents." if is_group_chat else "No relevant content found in this PDF."
                    st.error(error_msg)

def main():
    st.set_page_config(
        page_title="PDF Chat with Vector Database", 
        page_icon="📚", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    if not st.session_state.initialized:
        setup_page()
    elif st.session_state.current_page == 'setup':
        setup_page()
    elif st.session_state.current_page == 'main_menu':
        main_menu_page()
    elif st.session_state.current_page == 'pdf_selection':
        pdf_selection_page()
    elif st.session_state.current_page == 'group_management':
        group_management_page()
    elif st.session_state.current_page == 'single_chat' and st.session_state.selected_pdf:
        single_chat_page()
    elif st.session_state.current_page == 'group_chat' and st.session_state.selected_group:
        group_chat_page()
    else:
        if st.session_state.initialized:
            st.session_state.current_page = 'main_menu'
            main_menu_page()
        else:
            setup_page()

if __name__ == "__main__":
    main()