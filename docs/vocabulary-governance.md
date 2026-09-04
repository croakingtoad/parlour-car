## Controlled vocabulary governance

`vocabulary_terms` is a curator-owned vocabulary. Its durable states are
`proposed`, `canonical`, `deprecated`, and `merged`; a merged term records its
canonical target in `merged_into`. Merging does not retag thematic entries or
chunks. Review the reported match count and make any retagging decision
separately.

Index sections are deliberately not a vocabulary source. A book index carries
page locators, subentries, and cross-references, so treating each raw line as a
term produces noise rather than usable facets. Curators add reviewed candidates
through `manage_vocabulary`.

Cataloging performs a soft review only. When the vocabulary table exists, an
incoming `subject_headings` value that is not canonical emits a warning for
curation; the work is still cataloged. This preserves ingestion of legitimate
new subjects and does not automatically create a vocabulary proposal.
