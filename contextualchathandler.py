class ChatMemory:
    def __init__(self):
        self.chat_memory = []  # List to maintain chat history as {'user': ..., 'assistant': ...}

    def add_to_memory(self, user_query, assistant_response):
        """
        Add a query and its corresponding response to the chat memory.
        """
        self.chat_memory.append({"user": user_query, "assistant": assistant_response})

    def get_recent_memory(self, num_entries=3):
        """
        Retrieve the most recent chat entries.
        Returns formatted user-assistant pairs as a clear dialogue string.
        """
        recent_memory = self.chat_memory[-num_entries:]
        formatted_memory = "\n".join(
            [f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in recent_memory]
        )
        return formatted_memory
