from algebra.ff.gf2 import GF2

def test_basic_ops():
    assert GF2(0) + GF2(0) == GF2(0)
    assert GF2(0) + GF2(1) == GF2(1)
    assert GF2(1) + GF2(0) == GF2(1)
    assert GF2(1) + GF2(1) == GF2(0)
    
    assert GF2(0) * GF2(0) == GF2(0)
    assert GF2(0) * GF2(1) == GF2(0)
    assert GF2(1) * GF2(0) == GF2(0)
    assert GF2(1) * GF2(1) == GF2(1)

def test_polynomials():
    p_coeffs = [GF2(1), GF2(1)]  # 1 + x
    q_coeffs = [GF2(0), GF2(1)]  # x
    
    max_len = max(len(p_coeffs), len(q_coeffs))
    p_padded = p_coeffs + [GF2(0)] * (max_len - len(p_coeffs))
    q_padded = q_coeffs + [GF2(0)] * (max_len - len(q_coeffs))
    
    result = [p_padded[i] + q_padded[i] for i in range(max_len)]
    expected = [GF2(1), GF2(0)]  # 1 + 0x = 1
    assert result == expected
    
    p_mul_q = [GF2(0), GF2(1), GF2(1)]  # 0 + 1x + 1x^2 = x + x^2
    
    c0 = p_coeffs[0] * q_coeffs[0]  # 1 * 0 = 0
    c1 = p_coeffs[0] * q_coeffs[1] + p_coeffs[1] * q_coeffs[0]  # 1*1 + 1*0 = 1
    c2 = p_coeffs[1] * q_coeffs[1]  # 1 * 1 = 1
    
    assert [c0, c1, c2] == p_mul_q