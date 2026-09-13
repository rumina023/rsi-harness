"""Anchor Verifier prototype: decomposition, reconstruction and falsification.

This is an explicit reasoning scaffold, not a claim of human-level semantics.
It treats unknowns as first-class variables and never upgrades consistency to truth.
"""
from dataclasses import dataclass, asdict
import json
import re
from typing import Any


@dataclass
class Variable:
    name: str
    candidates: list[str]
    resolved: str | None = None
    confidence: float = 0.0


@dataclass
class MeaningGraph:
    source: str
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    variables: list[Variable]
    assumptions: list[str]


class AnchorVerifier:
    """Conservative verifier for short Japanese/English claims and policy changes."""
    def __init__(self, evidence: list[dict[str, Any]] | None = None):
        self.evidence = evidence or []

    def decompose(self, text: str) -> MeaningGraph:
        entities, relations, variables, assumptions = [], [], [], []
        # Lightweight lexicon; unknown terms remain explicit instead of being invented.
        words = re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9_.:-]+", text)
        known = {'私': 'agent', '僕': 'agent', 'I': 'agent', 'リンゴ': 'object', 'apple': 'object',
                 '大好き': 'attitude', '好き': 'attitude', '改善': 'change', '性能': 'metric',
                 '推論': 'capability', '記憶': 'capability', '探索': 'capability'}
        for word in words:
            if word in {'は', 'が', 'を', 'の', 'に', 'で', 'だ', 'です', 'する', 'is', 'a', 'the'}:
                continue
            kind = known.get(word, 'unknown')
            entity = {'id': f'e{len(entities)}', 'text': word, 'kind': kind, 'attributes': []}
            if word in ('赤い', '青い', '重い', '軽い', '甘い', '苦い'):
                entity['kind'] = 'attribute'
                entity['attributes'] = [{'name': 'value', 'value': word}]
            entities.append(entity)
            if kind == 'unknown':
                variables.append(Variable(f'{word}.identity', [word, 'unknown']))
                assumptions.append(f'{word} refers to a stable entity')
        # Detect simple subject-attitude-object structure.
        if len(entities) >= 2:
            subject, obj = entities[0], entities[-1]
            relation = 'likes' if any(x in text for x in ('好き', '大好き', 'likes', 'love')) else 'relates_to'
            relations.append({'subject': subject['id'], 'predicate': relation, 'object': obj['id'], 'confidence': .55})
        if any(x in text for x in ('大好き', 'love', 'likes')):
            variables.append(Variable('attitude.degree', ['low', 'medium', 'high'], None, .33))
            assumptions.append('attitude expression is interpreted on an ordinal scale')
        return MeaningGraph(text, entities, relations, variables, assumptions)

    def interventions(self, graph: MeaningGraph) -> list[dict[str, Any]]:
        """Generate one-variable interventions; execution is delegated to an evaluator."""
        tests=[]
        for entity in graph.entities:
            if entity['kind']=='attribute':
                value=entity['attributes'][0]['value']
                alternatives=[x for x in ('赤い','青い','重い','軽い','甘い','苦い') if x!=value]
                for alt in alternatives[:2]:
                    tests.append({'target':entity['id'],'change':{'value':alt},'expected':'結論への影響を再評価','status':'needs_experiment'})
        for relation in graph.relations:
            tests.append({'target':relation['predicate'],'change':{'enabled':False},'expected':'関係に依存する結論は変化','status':'needs_experiment'})
        return tests

    def reconstruct(self, graph: MeaningGraph) -> str:
        if graph.relations:
            r = graph.relations[0]
            lookup = {e['id']: e['text'] for e in graph.entities}
            pred = {'likes': '好き', 'relates_to': '関係する'}.get(r['predicate'], r['predicate'])
            return f"{lookup.get(r['subject'], '?')}は{lookup.get(r['object'], '?')}が{pred}"
        return ' '.join(e['text'] for e in graph.entities)

    def consistency(self, graph: MeaningGraph, reconstruction: str) -> dict[str, Any]:
        source_tokens = set(re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9_.:-]+", graph.source))
        rebuilt_tokens = set(re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9_.:-]+", reconstruction))
        preserved = len(source_tokens & rebuilt_tokens) / max(1, len(source_tokens))
        return {'token_preservation': preserved, 'relations_preserved': bool(graph.relations),
                'unknown_count': len(graph.variables), 'consistent': preserved >= .5 and bool(graph.entities)}

    def counterfactuals(self, graph: MeaningGraph) -> list[dict[str, Any]]:
        tests = []
        for var in graph.variables:
            for candidate in var.candidates:
                if candidate == var.resolved:
                    continue
                tests.append({'variable': var.name, 'alternative': candidate,
                              'question': f'If {var.name}={candidate}, does the conclusion still hold?',
                              'status': 'needs_experiment'})
        for relation in graph.relations:
            tests.append({'variable': 'relation', 'alternative': 'remove',
                          'question': f"What changes if {relation['predicate']} is false?",
                          'status': 'needs_experiment'})
        return tests

    def verify(self, text: str, claim: str | None = None) -> dict[str, Any]:
        graph = self.decompose(text)
        rebuilt = self.reconstruct(graph)
        check = self.consistency(graph, rebuilt)
        tests = self.counterfactuals(graph)
        interventions = self.interventions(graph)
        matched = [e for e in self.evidence if e.get('claim') == (claim or text)]
        contradicting = [e for e in matched if e.get('supports') is False]
        supporting = [e for e in matched if e.get('supports') is True]
        if contradicting:
            status = '反証あり'
        elif supporting and check['consistent']:
            status = '未反証'
        elif not check['consistent'] or tests:
            status = '証拠不足'
        else:
            status = '適用範囲外'
        return {'status': status, 'graph': asdict(graph), 'reconstruction': rebuilt,
                'consistency': check, 'falsification_tests': tests, 'interventions': interventions,
                'evidence': {'supporting': len(supporting), 'contradicting': len(contradicting)},
                'epistemic_note': '一貫性は真理を保証しない。反証されていない範囲だけを保持する。'}


def verify_json(text: str) -> str:
    return json.dumps(AnchorVerifier().verify(text), ensure_ascii=False, indent=2)


def verifier_benchmark() -> dict[str, Any]:
    """Small known-defect suite: consistency must not be confused with truth."""
    cases = [
        ('valid structure', '私は赤いリンゴが好きだ', False),
        ('unknown identity', 'ザルゴンは青いリンゴが好きだ', False),
        ('contradicted evidence', 'リンゴは青い', True),
    ]
    v=AnchorVerifier([{'claim':'リンゴは青い','supports':False}])
    results=[]
    for name,text,contradiction in cases:
        r=v.verify(text)
        results.append({'name':name,'status':r['status'],'has_interventions':bool(r['interventions']),
                        'passed': (r['status']=='反証あり')==contradiction})
    return {'cases':results,'pass_rate':sum(x['passed'] for x in results)/len(results),
            'principle':'再構成の成功を真理の証明として数えない'}


if __name__ == '__main__':
    import sys
    if sys.argv[1:] and sys.argv[1]=='--benchmark':
        print(json.dumps(verifier_benchmark(),ensure_ascii=False,indent=2))
    else:
        print(verify_json(' '.join(sys.argv[1:]) or '私はリンゴが大好きだ'))
