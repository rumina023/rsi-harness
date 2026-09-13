"""Connect Anchor Verifier to policy adoption without treating consistency as truth."""
from anchor_verifier import AnchorVerifier

def check(policy, difference):
    claim=(f"Policy changes explore={policy.get('explore')}, flip={policy.get('flip')}, "
           f"temperature={policy.get('temperature')}, restart={policy.get('restart')} improve search")
    evidence=[{'claim':claim,'supports': difference.get('delta',0)>0 and difference.get('interval',[0])[0]>0}]
    result=AnchorVerifier(evidence).verify(claim, claim)
    # The gate is conservative: contradictions block; unresolved variables are logged,
    # while empirical adoption still requires the fixed numerical policy gate.
    result['adoption_allowed'] = result['status'] != '反証あり'
    return result
