"""
Liste centralisée des tickers BRVM (actions + indices).
Source de vérité unique partagée entre les scrapers.

Actions : 48 sociétés cotées (source : https://www.sikafinance.com/marches/aaz)
  Note : SVOC.ci (Solibra) a une fiche société mais pas d'historique de cours
  exposé par l'API Sikafinance (toujours 'nodata').

Indices : 13 indices vivants sur Sikafinance (les anciens indices sectoriels
BRVMAG/BRVMAS/BRVMDI/BRVMFI/BRVMIN/BRVMTR/BRVMSP ont été remplacés par la
refonte sectorielle BRVM 2023 → BRVM-CB/CD/EN/IN/SF/SP/TEL).
"""

ACTIONS = [
    "ABJC.ci",
    "BICC.ci",
    "BICB.bj",
    "BNBC.ci",
    "BOAB.bj",
    "BOABF.bf",
    "BOAC.ci",
    "BOAM.ml",
    "BOAN.ne",
    "BOAS.sn",
    "CABC.ci",
    "CBIBF.bf",
    "CFAC.ci",
    "CIEC.ci",
    "ECOC.ci",
    "ETIT.tg",
    "FTSC.ci",
    "LNBB.bj",
    "NEIC.ci",
    "NSBC.ci",
    "NTLC.ci",
    "ONTBF.bf",
    "ORAC.ci",
    "ORGT.tg",
    "PALC.ci",
    "PRSC.ci",
    "SAFC.ci",
    "SCRC.ci",
    "SDCC.ci",
    "SDSC.ci",
    "SEMC.ci",
    "SGBC.ci",
    "SHEC.ci",
    "SIBC.ci",
    "SICC.ci",
    "SIVC.ci",
    "SLBC.ci",
    "SMBC.ci",
    "SNTS.sn",
    "SOGC.ci",
    "SPHC.ci",
    "STAC.ci",
    "STBC.ci",
    "SVOC.ci",
    "TTLC.ci",
    "TTLS.sn",
    "UNLC.ci",
    "UNXC.ci",
]

INDICES = [
    "BRVMC",
    "BRVM30",
    "BRVMPR",
    "BRVMPA",
    "BRVM-CB",
    "BRVM-CD",
    "BRVM-EN",
    "BRVM-IN",
    "BRVM-SF",
    "BRVM-TEL",
    "CAPIBRVM",
    "SIKAIDX",
    "SIKATR",
]


def safe_filename(ticker: str) -> str:
    """Convertit un ticker en nom de fichier sûr (ex: BRVM-CB -> BRVM_CB)."""
    return ticker.replace(".", "_").replace("-", "_")
