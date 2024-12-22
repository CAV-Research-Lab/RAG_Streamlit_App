import chromadb
import openai
import os
import streamlit as st
 

class Response:
    
    def __init__(self, chat_handler):
        self.query_collection = {}
        current_dir = os.path.dirname(__file__)
        chromadb_dir = os.path.join(current_dir, "chroma_db")
        self.client = chromadb.PersistentClient(chromadb_dir) 
        self.collection = self.client.get_collection("RAG_Assistant")
        self.chat_handler = chat_handler  # Use shared ChatMemory instance
        self.openai_api_key =  st.secrets["OPENAI_API_KEY"]

        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.")
        
    def content_retrieval(self, query, n_results=3):
        # Retrieve the collection
        results = self.collection.query(query_texts=query, n_results=n_results)
        return results


    def get_completion(self, user_prompt, system_prompt, model="gpt-3.5-turbo"):
        openai.api_key = self.openai_api_key
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    
    def rewrite_query(self, query, chat_history):
        """
        Rewrite the user's latest query to include relevant context from the chat history.
        """
        prompt = f"""
        You are a highly intelligent assistant helping a student in an ongoing conversation about intelligent vehicle design.
        Your task is to refine the user's latest query so that it incorporates relevant context from the previous conversation
        and provides clarity for follow-up questions.

        ### Conversation History:
        {chat_history}

        ### User's Latest Query:
        {query}

        ### Instructions:
        1. Combine User's Latest Query with details from Conversation History to rewrite it in a complete and contextually aware form.
        2. Be explicit and clear. Retain continuity by ensuring that the rewritten query relates to the previous discussion.
        3. Do NOT ask for further clarifications in the rewritten query. Assume sufficient context has already been provided.

        ### Now rewrite the user's query:
        Rewritten Query:"""

        system_prompt = self.make_system_prompt()
        response = self.get_completion(user_prompt=prompt, system_prompt=system_prompt)
        return response.split("Rewritten Query:")[1].strip() if "Rewritten Query:" in response else query


    def make_hyde_prompt(self, query):
        return f"""
        ### INSTRUCTIONS
        Generate a plausible hypothetical document that answers the QUERY below. Use language and technical terminology relevant to "intelligent vehicle design". 

        - Ensure the response is specific, focused, and concise (around 3-4 sentences). 
        - Use terminology and phrases that align with academic or technical contexts.

        ### QUERY
        {query}
        """




    def expand_search_results(self, search_results):
        """
        Expand search results into a formatted string, including metadata for citations.
        """
        try:
            if not search_results or not isinstance(search_results, dict):
                raise ValueError("Invalid search results format. Must be a dictionary.")

            result_str = ""

            # Loop through the search results and format each result
            documents = search_results.get("documents", [[]])[0]
            metadatas = search_results.get("metadatas", [[]])[0]

            if not documents:
                return "No documents found in the search results."

            for doc, metadata in zip(documents, metadatas):
                title = metadata.get("title", "Unknown Title")
                author = metadata.get("author", "Unknown Author")
                year = metadata.get("year", "Unknown Year")

                result_str += f"\n[Title: {title}, Author: {author}, Year: {year}]\n{doc.strip()}\n"

            return result_str.strip()

        except Exception as e:
            print(f"Error expanding search results: {e}")
            return "Error expanding search results."
        
        
    def get_RAG_completion(self, query, n_results=5):
        """
        Generate a context-aware response by incorporating the chat history and search results.
        """
        # Build the chat history as a string
        chat_history_list = self.chat_handler.get_recent_memory(num_entries=5)  # Get recent 5 queries
        chat_history = "\n".join(chat_history_list)  # Combine into a single string

        # Rewrite the query with the context of chat history
        refined_query = self.rewrite_query(query, chat_history)

        # Generate hypothetical document embedding
        hyde_prompt = self.make_hyde_prompt(refined_query)
        hyde_query = self.get_completion(hyde_prompt, system_prompt=self.make_system_prompt())

        # Retrieve search results
        search_results = self.content_retrieval(hyde_query)
        result_str = self.expand_search_results(search_results)

        # Generate the RAG prompt
        rag_prompt = self.make_rag_prompt(refined_query, result_str)

        # Call get_completion with the RAG prompt
        return self.get_completion(user_prompt=rag_prompt, system_prompt=self.make_system_prompt())




    def make_rag_prompt(self, query, result_str):
        return f"""
        ### QUERY:
        {query}

        ### CONTEXT:
        {result_str}

        ### INSTRUCTIONS:
        1. Always prioritise the provided lecture materials, syllabus, and approved course documents retrieved from the knowledge collection as the primary sources of information when answering questions.
        2. Use the QUERY for intelligent vehicles to accurately understand the student's intent and retrieve the  CONTEXT from the collection. Ensure that your response directly addresses the QUERY.
        3. When referencing the  CONTEXT content, integrate it seamlessly into your response, citing the source explicitly to maintain transparency and credibility.
        4. If the  CONTEXT from the collection do not fully answer the query, complement them with trusted external resources, clearly indicating their origin.
        5. Structure your response to ensure clarity, conciseness, and alignment with academic standards, avoiding speculation or unsubstantiated information.
        6. If no relevant information is retrieved from the collection or trusted resources, explain this to the user and provide suggestions for alternative ways to explore the topic.
        7. Avoid directly solving assignments, exams, or graded work. Instead, provide insights, explanations, or references that guide the user towards understanding and independent problem-solving.
        8. Use the  CONTEXT to enrich your response with context, examples, or deeper insights that support the student’s learning experience.
        9. Handle ambiguous or unclear QUERY inputs by asking clarifying questions before providing a response.
        10. Always consider ethical guidelines, inclusivity, and the safety of the information shared in your responses. Ensure that your advice aligns with the university’s academic integrity standards.
        """

    def make_system_prompt(self):
        return """
        You are a reliable and knowledgeable **virtual teaching assistant** for university students enrolled in the **Intelligent Vehicle Design** module. Your role is to provide accurate, concise, and academically rigorous responses based on the provided course materials, syllabus, and approved resources.

        ### GUIDELINES:
        1. Answer student questions using an academic tone, prioritising clarity and relevance.
        2. Base your responses on the provided resources, avoiding speculation or personal opinions.
        3. Use trusted external sources only when the provided materials lack sufficient information. Cite these sources clearly.
        4. Encourage independent problem-solving by guiding students, rather than directly solving assignments or exams.
        5. Maintain a professional structure in your responses, suitable for academic communication.
        6. Prioritise inclusivity, ethical considerations, and safety in all answers.
        7. If a query is ambiguous, seek clarification to ensure a precise response.
        8. When unable to answer a question, explain why and suggest alternative resources for further exploration.

    Your primary focus is to enhance students’ understanding and foster independent learning, while strictly adhering to academic standards.
    """