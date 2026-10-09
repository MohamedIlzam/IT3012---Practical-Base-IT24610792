# logic_engine.py
"""
Logic Engine implementing a Propositional Horn-Clause Knowledge Base
with Forward Chaining for SE3062 Intelligent Systems Practical 07.
"""
from typing import List, Tuple, Set


class KnowledgeBase:
    """A Horn-clause Knowledge Base supporting forward chaining inference."""

    def __init__(self):
        # self.facts must be a set of unique string facts
        self.facts: Set[str] = set()
        # self.rules must be a list of tuples (premise_list, conclusion_string)
        self.rules: List[Tuple[List[str], str]] = []

    def tell_fact(self, fact_string: str) -> None:
        """Add a factual assertion to the knowledge base."""
        self.facts.add(str(fact_string))

    def tell_rule(self, premise_list: List[str], conclusion_string: str) -> None:
        """Register a Horn-clause rule: premises => conclusion."""
        self.rules.append((list(premise_list), str(conclusion_string)))

    def clear_facts(self) -> None:
        """Remove all facts while preserving registered rules."""
        self.facts.clear()

    def forward_chain(self) -> None:
        """
        Forward chaining inference algorithm.
        Repeats rule evaluation in a while loop until a full pass generates no new facts.
        For each rule, infer the conclusion only if every premise is in self.facts
        and the conclusion is not already present.
        """
        while True:
            new_fact_added = False
            for premises, conclusion in self.rules:
                # Check if every premise is in self.facts
                if all(premise in self.facts for premise in premises):
                    # Infer conclusion only if not already present
                    if conclusion not in self.facts:
                        self.facts.add(conclusion)
                        new_fact_added = True
            if not new_fact_added:
                break

    def ask(self, query: str) -> bool:
        """Check if a query fact is present in the knowledge base."""
        return query in self.facts
