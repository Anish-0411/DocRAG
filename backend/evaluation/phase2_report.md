# CortexRAG Phase 2 — 40-Question Evaluation

## Dataset status

- Total questions: 40
- Answers present: 40/40
- Questions with at least one answer citation: 38/40
- Retrieved evidence entries: 200
- Evidence entries marked cited: 51

## Automatic structural metrics

| Metric | Result |
|---|---:|
| Answer presence | 100.0% |
| Answers containing at least one citation | 95.0% |
| Citation IDs found in retrieved evidence | 100.0% |
| Citation IDs consistent with cited flags | 97.5% |
| Retrieved evidence marked cited / total retrieved | 25.5% |
| Exact gold chunk match | 60.0% |
| Gold document match | 80.0% |
| Gold page match | 72.5% |

These structural checks do not establish factual correctness or semantic citation support.

## Manual answer-quality evaluation

Scores range from 0 to 2. Pending questions are excluded from averages.

| Dimension | Evaluated | Pending | Mean | Score 2 rate | Score 0 | Score 1 | Score 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| correctness | 40 | 0 | 1.675 | 80.0% | 5 | 3 | 32 |
| faithfulness | 39 | 1 | 2 | 100.0% | 0 | 0 | 39 |
| answer_relevance | 40 | 0 | 1.8 | 90.0% | 4 | 0 | 36 |
| citation_support | 40 | 0 | 1.9 | 92.5% | 1 | 2 | 37 |


## Abstention evaluation

- Evaluated abstention cases: 1
- Appropriate abstentions: 0
- Appropriate abstention rate: 0.0%

Questions where the system attempted an answer are excluded.

## Structural and manual diagnostic counts

These categories are diagnostic signals, not confirmed root causes.
They may overlap and must not be added together.

| Diagnostic | Questions |
|---|---:|
| no_answer | 0 |
| no_answer_citation | 2 |
| invalid_citation_reference | 0 |
| citation_flag_inconsistency | 1 |
| exact_gold_chunk_not_cited | 16 |
| gold_document_not_cited | 8 |
| gold_page_not_cited | 11 |
| manually_scored_correctness_below_2 | 8 |
| inappropriate_abstention | 1 |


## Question-level evaluation

| Question | Correctness | Faithfulness | Relevance | Citation support | Notes |
|---|---:|---:|---:|---:|---|
| Q01 | 2 | 2 | 2 | 2 | Correct answer. Exact gold chunk and citation support the location. |
| Q02 | 0 | 2 | 0 | 2 | Wrong aircraft accident report retrieved. Answer is supported by cited evidence, but not by the evidence relevant to this question. |
| Q03 | 0 | 2 | 0 | 2 | Wrong aircraft accident report retrieved. Answer is supported by cited evidence, but not by the evidence relevant to this question. |
| Q04 | 0 | 2 | 0 | 2 | Wrong aircraft accident report retrieved. Answer is supported by cited evidence, but not by the evidence relevant to this question. |
| Q05 | 2 | 2 | 2 | 2 | Correct Login.gov explanation supported by the cited annual report passage. |
| Q06 | 1 | 2 | 2 | 2 | Partial answer. Cited passage supports the 90% infection-reduction and 65% mortality-reduction targets, but the answer omits the diagnosis and treatment targets requested. |
| Q07 | 0 | Pending | 0 | 0 | Inappropriate abstention. Relevant gold evidence exists in the indexed corpus, but the system did not provide an answer or citation. Faithfulness is N/A because the system abstained and made no factual claims. |
| Q08 | 2 | 2 | 2 | 2 | Correct explanation of hybrid retrieval, supported by the cited evidence. |
| Q09 | 2 | 2 | 2 | 2 | Correct five curriculum areas, supported by the cited evidence. |
| Q10 | 2 | 2 | 2 | 2 | Correct explanation of standards and employment/firm growth, supported by the cited evidence. |
| Q11 | 2 | 2 | 2 | 2 | Correct: nearly 90% of global trade. Directly supported by the cited report passage. |
| Q12 | 2 | 2 | 2 | 2 | Correct: more open trade regime and increasing foreign direct investment. Directly supported by the cited passage. |
| Q13 | 2 | 2 | 2 | 2 | Correct: annual U.S. emissions and removals by source, economic sector, and greenhouse gas. Exact semantic match. |
| Q14 | 2 | 2 | 2 | 2 | Correct: all listed climate impacts are included. Both cited EPA passages jointly support the complete answer. |
| Q15 | 2 | 2 | 2 | 2 | Correct: 20 Climate Change Adaptation Implementation Plans. Directly supported by both cited passages. |
| Q16 | 2 | 2 | 2 | 2 | Correct. Includes children and people with preexisting respiratory conditions, plus additional populations supported by the cited EPA air-quality strategy passages. |
| Q17 | 2 | 2 | 2 | 2 | Correct. FAO and UNFCCC are explicitly named in the cited EPA passage. |
| Q18 | 2 | 2 | 2 | 2 | Correct. 21% diagnosed and 13% treated in 2019, directly supported by the cited HCV protocol. |
| Q19 | 2 | 2 | 2 | 2 | Correct. SVR rates greater than 95% after 8 to 12 weeks, directly supported by the cited passage. |
| Q20 | 2 | 2 | 2 | 2 | Correct. All four barriers—professional knowledge, screening-test availability/access, stigma, and discrimination—are supported by the cited passage. |
| Q21 | 2 | 2 | 2 | 2 | Correct: 878 HCV cases reported in Colombia in 2021. Exact gold evidence is cited. |
| Q22 | 2 | 2 | 2 | 1 | Correct explanation of centralized DAA purchasing, access guarantees, cost reduction, and expansion to the subsidized regime. Citation marker EVD-F3E11AB23CCF points to supporting evidence, but that evidence is marked cited=false and is missing from the citations array; the other citation is unrelated. |
| Q23 | 2 | 2 | 2 | 2 | Correct: pilot total flight experience is 2,319 hours. The cited passage directly supports the answer. |
| Q24 | 2 | 2 | 2 | 1 | Correct runway dimensions: 1,500 ft by 75 ft. The answer includes an unnecessary second runway detail from another accident report; citation support is therefore partial. |
| Q25 | 0 | 2 | 2 | 2 | Incorrect for the intended flight: expected Quincy, Florida (2J9), to Tallahassee, Florida; generated answer describes Wausau, Wisconsin. The answer is faithful to the retrieved but unrelated passage. |
| Q26 | 2 | 2 | 2 | 2 | Correct: no pre-impact mechanical malfunction was identified. The cited passage directly confirms the wreckage examination found no such malfunction and the pilot reported none. |
| Q27 | 2 | 2 | 2 | 2 | Correct: all five NTSB responsibilities are accurately stated and directly supported by the cited passage. Exact gold chunk ID was not retrieved, but an equivalent passage from another accident report was retrieved. This is an exact-chunk evaluation limitation, not a substantive retrieval failure. |
| Q28 | 2 | 2 | 2 | 2 | All five critical-infrastructure sectors are correct. Citation points to the exact gold chunk and supports the complete answer. |
| Q29 | 2 | 2 | 2 | 2 | Correct: more than $6 trillion annually by 2021. Citation points to the exact gold chunk and directly supports the answer. |
| Q30 | 2 | 2 | 2 | 2 | Correct and sufficiently supported. Citations cover tools, training, guidelines, vulnerability and resilience assessments, intelligence sharing, and incident response. |
| Q31 | 2 | 2 | 2 | 2 | Correctly identifies USAJOBS and Retirement Services. The cited passage directly supports both. |
| Q32 | 2 | 2 | 2 | 2 | Correctly identifies the 4.3 out of 5 target. The cited evidence directly supports the answer, although its cited flag is false despite being referenced in the answer. Track as a citation-flag consistency issue. |
| Q33 | 2 | 2 | 2 | 2 | Correctly identifies the Retirement Services FY2026 target of 4.2 out of 5. Directly supported by the cited passage. |
| Q34 | 2 | 2 | 2 | 2 | All three EPA FY2022-2023 Agency Priority Goals are accurately stated and directly supported by the cited passage. |
| Q35 | 2 | 2 | 2 | 2 | Correctly identifies low-income communities and communities of color. Directly supported by the cited passage. |
| Q36 | 1 | 2 | 2 | 2 | Partially correct. The answer describes GHG regulation, reporting, and partnership programmes, supported by citations, but misses the expected emphasis on energy/resource efficiency, renewable energy, and decarbonizing buildings, transportation, power, and industry. |
| Q37 | 1 | 2 | 2 | 2 | Partially correct. Technical expertise, guidance, capacity-building, and climate assistance are supported. The answer does not explicitly cover learning from other countries or supporting science-based international mitigation and adaptation goals. |
| Q38 | 2 | 2 | 2 | 2 | Correctly covers lower trade costs, higher productivity, higher-value exports, and comparative advantages. Additional claims are supported by the second citation. |
| Q39 | 2 | 2 | 2 | 2 | Correctly covers exports, consumer protection, innovation, and environmental sustainability. Directly supported by the cited passage. |
| Q40 | 2 | 2 | 2 | 2 | Correctly explains flexibility and adaptability without rigid compliance requirements, supporting innovation efficiency in uncertain markets. Directly supported by the citation. |


## Interpretation notes

- Citation ID validity checks structure, not factual support.
- Exact gold chunk matching is a strict evidence-alignment proxy.
- A different chunk may still contain semantically valid evidence.
- Document and page matches do not independently prove faithfulness.
- Human scores are manual judgments, not model-generated metrics.
- Pending manual scores are excluded from averages.
- Diagnostic categories can overlap.
- This evaluator performs no model or external API calls.
