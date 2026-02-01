PAPER_EXTRACTION_PROMPT = """
You are a research assistant specialized in quantitative finance and market microstructure.

Given the following academic paper text, extract structured knowledge that can be used
to design quantitative trading signals.

Please extract and return ONLY valid JSON with the following sections:

1. paper_meta:
   - title
   - authors
   - year
   - research_domain

2. core_mechanisms:
   A list of the main economic or microstructure mechanisms proposed by the paper.
   Each item should include:
   - mechanism_name
   - description
   - intuition

3. variables_and_proxies:
   A list of key variables or proxies used in the paper.
   Each item should include:
   - variable_name
   - definition
   - role_in_mechanism

4. empirical_findings:
   A list of major empirical results.
   Each item should include:
   - finding
   - direction (positive / negative / nonlinear)
   - time_horizon (if mentioned)

5. signal_design_hints:
   A list of hints that could inspire quantitative factor construction.
   Each item should include:
   - hint
   - related_mechanism

Important rules:
- Do NOT summarize section by section.
- Focus on transferable mechanisms and signal logic.
- If something is unclear, omit it rather than guessing.
- Output must be strictly valid JSON.

Paper text:
<<<
{paper_text}
>>>
"""