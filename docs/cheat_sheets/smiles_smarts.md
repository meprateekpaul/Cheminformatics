# SMILES and SMARTS Quick Reference

## SMILES (Simplified Molecular Input Line Entry System)

### Atoms
- Organic subset (implicit hydrogens allowed): `B, C, N, O, P, S, F, Cl, Br, I`
- Aromatic atoms: lowercase `c, n, o, p, s`
- Bracket atoms (explicit hydrogens, isotopes, charges): `[NH4+]`, `[O-]`, `[13CH4]`, `[Fe+2]`

### Bonds
- Single bond: `-` (usually omitted)
- Double bond: `=`
- Triple bond: `#`
- Aromatic bond: `:` (usually omitted)
- Disconnected structures (salts/solvents): `.`

### Branching & Rings
- Branches enclosed in parentheses: `CC(=O)O` (acetic acid)
- Nested branches: `CCN(CC)CC` (triethylamine)
- Ring closures: single digit `1` to `9` (e.g. `c1ccccc1` for benzene)
- Multi-digit rings: `%10` to `%99` (e.g. `C%10...C%10`)

### Stereochemistry
- Tetrahedral chiral center: `@` (anticlockwise / counterclockwise) and `@@` (clockwise)
- Cis/trans double bonds: `/` and `\` direction flags (e.g. `F/C=C/F` trans, `F/C=C\F` cis)

---

## SMARTS (SMILES Arbitrary Target Specification)

### Atomic Primitives
- `[#6]`: Any Carbon atom (atomic number 6)
- `[#7]`: Any Nitrogen atom
- `[#8]`: Any Oxygen atom
- `[!#6]`: Any non-carbon (heteroatom)
- `[a]`: Any aromatic atom
- `[A]`: Any aliphatic atom
- `[D<n>]`: Atom with degree $n$ (number of explicit heavy bonds)
- `[H<n>]`: Atom with $n$ attached hydrogens
- `[v<n>]`: Atom with total valence $n$
- `[X<n>]`: Atom with connectivity $n$ (heavy + hydrogen)
- `[+<n>]` / `[-<n>]`: Formal charge

### Common Functional Group SMARTS
| Functional Group | SMARTS Pattern |
|---|---|
| Carboxylic acid | `[CX3](=O)[OX2H1]` |
| Ester | `[#6][CX3](=O)[OX2H0][#6]` |
| Amide | `[#6][CX3](=O)[NX3H0,NX3H1,NX3H2]` |
| Primary amine | `[NX3H2;!$(NC=O)]` |
| Secondary amine | `[NX3H1;!$(NC=O)]([#6])[#6]` |
| Tertiary amine | `[NX3H0;!$(NC=O)]([#6])([#6])[#6]` |
| Alcohol | `[OX2H][CX4]` |
| Phenol | `[OX2H][c]` |
| Ether | `[OD2]([#6])[#6]` |
| Aldehyde | `[CX3H1](=O)[#6,H]` |
| Ketone | `[#6][CX3](=O)[#6]` |
| Nitrile | `[NX1]#[CX2]` |
| Nitro | `[$([NX3](=O)=O),$([NX3+](=O)[O-])]` |
| Sulfonamide | `[SX4](=O)(=O)[NX3]` |
| Halogen | `[F,Cl,Br,I]` |
