import requests
from django.shortcuts import render

# URL of your Flask RAG model deployed on Render
URL = "https://sih-25.onrender.com/ask"

def query_rag(request):
    response_data = None
    error_message = None

    if request.method == "POST":
        user_query = request.POST.get("query")
        try:
            # Send the query to Flask RAG API
            resp = requests.post(URL, json={"query": user_query})
            if resp.status_code == 200:
                response_data = resp.json().get("answer", "No response field found")
            else:
                error_message = f"Error: Flask API returned {resp.status_code}"
        except Exception as e:
            error_message = f"Request failed: {str(e)}"

    return render(request, "query_rag.html", {
        "response": response_data,
        "error": error_message
    })
