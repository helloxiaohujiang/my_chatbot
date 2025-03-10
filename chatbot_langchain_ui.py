import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global memeory to persist conversations
memory = ConversationBufferMemory()

# Load TinyLlama
llm = LlamaCpp(
    model_path = './mistral-7b-instruct-v0.2.Q4_K_S.gguf',  
    n_ctx=2048,  
    temperature=0.1,
    n_batch=512,  
    n_gpu_layers=0,
    max_tokens=256,
    stop=["\n", ".", "?", "!"]
  
)

# Attach persistent memory
chatbot = ConversationChain(llm=llm, memory=memory)

# Dash Layout
app.layout = dbc.Container([
    html.H1("💬 AI Chatbot", className="text-center"),
    
    # Chat History Display
    dcc.Markdown(id="chat-history", 
                    style={
        "white-space": "pre-wrap",  # Keeps the line breaks
        "border": "2px solid #ccc",  # Adds a solid border
        "border-radius": "8px",  # Optional: rounded corners for the border
        "padding": "10px",  # Adds padding inside the box
        "background-color": "#f9f9f9",  # Background color of the box
        "max-height": "400px",  # Optional: sets a maximum height for scrolling
        "overflow-y": "auto",  # Allows vertical scrolling if the content overflows
    }),

    # User Input
    dbc.Input(id="user-input", type="text", placeholder="Type a message...", debounce=True),
    
    # Send Button
    dbc.Button("Send", id="send-button", color="primary", className="mt-2"),
    
    # Hidden Store for Chat History
    dcc.Store(id="chat-store", data={"history": []})
], fluid=True)

# Callback for chatbot response
@app.callback(
    [Output("chat-history", "children"), Output("chat-store", "data")],
    [Input("send-button", "n_clicks")],
    [State("user-input", "value"), State("chat-store", "data")]
)
def update_chat(n_clicks, user_input, chat_data):
    if not user_input:
        return dash.no_update  # Don't update if input is empty
    
    # Get chatbot response
    response = chatbot.predict(input=user_input)
    
    if response and response[-1] not in ".!?":
        response += "."

    # Update history
    chat_data["history"].append(f"**User:** {user_input}\n**AI:** {response}\n")
    
    # Keep only last 5 messages
    chat_data["history"] = chat_data["history"][-5:]

    # Update displayed history
    chat_display = "\n".join(chat_data["history"])
    
    return chat_display, chat_data

if __name__ == "__main__":
    app.run_server(debug=True, port=8080)