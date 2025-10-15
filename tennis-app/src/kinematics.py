import numpy as np

def angle(a, b, c):
    a, b, c = map(np.asarray, (a, b, c))
    v1, v2 = a - b, c - b
    denom = (np.linalg.norm(v1)*np.linalg.norm(v2)) + 1e-9
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def body_height(landmarks):
    # use nose (0) to ankle (27 or 28) as a proxy; fall back to shoulder-ankle
    nose = landmarks[0][:2]
    r_ank = landmarks[28][:2]
    return np.linalg.norm(r_ank - nose)
