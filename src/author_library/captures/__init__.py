"""Chrome extension capture processing for Parlour Car.

Handles capture events from Parlour Chrome: validates payloads, queues
background processing via arq, fetches/caches YouTube transcripts,
generates source overviews, and processes quick/deep/visual captures.
"""
