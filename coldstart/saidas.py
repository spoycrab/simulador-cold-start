"""Gravação de tabelas em CSV e em LaTeX (modelo SBC)."""

from __future__ import annotations

import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

_ESPECIAIS_LATEX = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escapar_latex(texto: str) -> str:
    return "".join(_ESPECIAIS_LATEX.get(c, c) for c in str(texto))


def formatar_numero(valor, casas: int = 2) -> str:
    """Formata números no padrão brasileiro (vírgula decimal, ponto de milhar)."""
    if valor is None:
        return "--"
    if isinstance(valor, (bool, np.bool_)):
        return "Sim" if valor else "Não"
    if isinstance(valor, numbers.Real) and math.isnan(float(valor)):
        return "--"
    if isinstance(valor, numbers.Integral) or (
        isinstance(valor, numbers.Real) and float(valor).is_integer() and casas == 0
    ):
        texto = f"{int(valor):,}"
    elif isinstance(valor, numbers.Real):
        texto = f"{float(valor):,.{casas}f}"
    else:
        return str(valor)
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def tabela_latex(df: pd.DataFrame, legenda: str, rotulo: str, casas: int = 2) -> str:
    """Gera um ambiente table/tabular simples, compatível com o modelo SBC."""
    alinhamento = "l" + "r" * (len(df.columns) - 1)
    linhas = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{escapar_latex(legenda)}}}",
        rf"\label{{{rotulo}}}",
        rf"\begin{{tabular}}{{{alinhamento}}}",
        r"\hline",
        " & ".join(rf"\textbf{{{escapar_latex(c)}}}" for c in df.columns) + r" \\",
        r"\hline",
    ]
    for _, registro in df.iterrows():
        celulas = []
        for valor in registro:
            if isinstance(valor, (numbers.Real, np.bool_)) or valor is None:
                celulas.append(formatar_numero(valor, casas))
            else:
                celulas.append(escapar_latex(valor))
        linhas.append(" & ".join(celulas) + r" \\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas) + "\n"


def salvar_tabela(df: pd.DataFrame, pasta: str | Path, nome: str, legenda: str,
                  rotulo: str | None = None, casas: int = 2) -> None:
    """Grava `nome`.csv (dados brutos) e `nome`.tex (tabela formatada)."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    df.to_csv(pasta / f"{nome}.csv", index=False, encoding="utf-8")
    texto = tabela_latex(df, legenda, rotulo or f"tab:{nome}", casas)
    (pasta / f"{nome}.tex").write_text(texto, encoding="utf-8")
