import math
import numpy as np


def solve_star(pL, rL, uL, pR, rR, uR, g=1.4):
    aL = math.sqrt(g * pL / rL)
    aR = math.sqrt(g * pR / rR)

    def f(p, pk, rk, ak):
        if p > pk:
            A = 2 / ((g + 1) * rk)
            B = (g - 1) / (g + 1) * pk
            return (p - pk) * math.sqrt(A / (p + B))
        return 2 * ak / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

    def fp(p, pk, rk, ak):
        if p > pk:
            A = 2 / ((g + 1) * rk)
            B = (g - 1) / (g + 1) * pk
            return math.sqrt(A / (B + p)) * (1 - (p - pk) / (2 * (B + p)))
        return 1 / (rk * ak) * (p / pk) ** (-(g + 1) / (2 * g))

    p = max(1e-6, ((aL + aR - 0.5 * (g - 1) * (uR - uL)) /
                   (aL / pL ** ((g - 1) / (2 * g)) + aR / pR ** ((g - 1) / (2 * g)))) ** (2 * g / (g - 1)))
    for _ in range(200):
        F = f(p, pL, rL, aL) + f(p, pR, rR, aR) + (uR - uL)
        FP = fp(p, pL, rL, aL) + fp(p, pR, rR, aR)
        dp = F / FP
        p = p - dp
        if p <= 0:
            p = 1e-8
        if abs(dp) / max(abs(p), 1e-30) < 1e-10:
            break
    ustar = 0.5 * (uL + uR) + 0.5 * (f(p, pR, rR, aR) - f(p, pL, rL, aL))
    return p, ustar, aL, aR


def sample_vec(xi, pL, rL, uL, pR, rR, uR, g=1.4):
    ps, us, aL, aR = solve_star(pL, rL, uL, pR, rR, uR, g)
    xi = np.asarray(xi, float)
    r = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)
    left = xi <= us
    Lm = left
    if ps > pL:
        SL = uL - aL * math.sqrt((g + 1) / (2 * g) * ps / pL + (g - 1) / (2 * g))
        undist = Lm & (xi <= SL)
        star = Lm & (xi > SL)
        r[undist] = rL; u[undist] = uL; p[undist] = pL
        rs = rL * ((ps / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pL + 1))
        r[star] = rs; u[star] = us; p[star] = ps
    else:
        SHL = uL - aL
        aLs = aL * (ps / pL) ** ((g - 1) / (2 * g))
        STL = us - aLs
        undist = Lm & (xi <= SHL)
        fan = Lm & (xi > SHL) & (xi < STL)
        star = Lm & (xi >= STL)
        r[undist] = rL; u[undist] = uL; p[undist] = pL
        xf = xi[fan]
        uf = 2 / (g + 1) * (aL + (g - 1) / 2 * uL + xf)
        af = 2 / (g + 1) * (aL + (g - 1) / 2 * (uL - xf))
        r[fan] = rL * (af / aL) ** (2 / (g - 1)); u[fan] = uf; p[fan] = pL * (af / aL) ** (2 * g / (g - 1))
        r[star] = rL * (ps / pL) ** (1 / g); u[star] = us; p[star] = ps
    Rm = ~left
    if ps > pR:
        SR = uR + aR * math.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g))
        undist = Rm & (xi >= SR)
        star = Rm & (xi < SR)
        r[undist] = rR; u[undist] = uR; p[undist] = pR
        rs = rR * ((ps / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pR + 1))
        r[star] = rs; u[star] = us; p[star] = ps
    else:
        SHR = uR + aR
        aRs = aR * (ps / pR) ** ((g - 1) / (2 * g))
        STR = us + aRs
        undist = Rm & (xi >= SHR)
        fan = Rm & (xi < SHR) & (xi > STR)
        star = Rm & (xi <= STR)
        r[undist] = rR; u[undist] = uR; p[undist] = pR
        xf = xi[fan]
        uf = 2 / (g + 1) * (-aR + (g - 1) / 2 * uR + xf)
        af = 2 / (g + 1) * (aR - (g - 1) / 2 * (uR - xf))
        r[fan] = rR * (af / aR) ** (2 / (g - 1)); u[fan] = uf; p[fan] = pR * (af / aR) ** (2 * g / (g - 1))
        r[star] = rR * (ps / pR) ** (1 / g); u[star] = us; p[star] = ps
    return r, u, p

