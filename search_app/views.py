# search_app/views.py

from django.shortcuts import render
from .bm25 import search_bm25  # Import your BM25 logic from the previous code

def search_view(request):
    query = request.GET.get('query', '')
    results = []

    if query:
        results = search_bm25(query)  # Call your search function here

    return render(request, 'search_app/search.html', {'query': query, 'results': results})
