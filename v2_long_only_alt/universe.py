"""
Univers STOXX Europe 600 — sélection des plus liquides avec tickers Yahoo Finance.
On vise ~250 actions avec historique long pour avoir un bon échantillon cross-sectional.
Note : utiliser la composition actuelle introduit un léger survivorship bias.
Pour un travail rigoureux, ré-importer l'historique des compositions (Bloomberg/Refinitiv).
"""

# Liste curated de ~250 grandes capitalisations européennes avec historique disponible
STOXX_600_TICKERS = [
    # === ROYAUME-UNI (.L) ===
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "BATS.L",
    "DGE.L", "REL.L", "LSEG.L", "AAL.L", "BARC.L", "NG.L", "VOD.L", "PRU.L",
    "TSCO.L", "GLEN.L", "LLOY.L", "IMB.L", "RKT.L", "EXPN.L", "STAN.L", "CRH.L",
    "BT-A.L", "CNA.L", "WTB.L", "ANTO.L", "WPP.L", "HLN.L", "LGEN.L", "AHT.L",
    "SSE.L", "JD.L", "SVT.L", "AV.L", "MNDI.L", "PSON.L", "BNZL.L", "INF.L",
    "ABF.L", "FRES.L", "BKG.L", "ADM.L", "RTO.L", "SGE.L", "SDR.L", "PSN.L",
    "ITRK.L", "MKS.L", "SMIN.L", "SN.L", "BA.L", "BME.L", "AUTO.L", "DPLM.L",
    "BAB.L", "RMV.L", "HLMA.L", "TW.L", "WEIR.L", "MRO.L", "PHNX.L", "CCH.L",
    "IAG.L", "III.L", "LAND.L", "SBRY.L", "ENT.L", "SGRO.L", "NXT.L", "BRBY.L",
    "MNG.L", "SPX.L", "TPK.L", "RR.L", "UU.L", "BDEV.L", "ICP.L", "BTRW.L",

    # === FRANCE (.PA) ===
    "MC.PA", "OR.PA", "RMS.PA", "TTE.PA", "SAN.PA", "AIR.PA", "BNP.PA", "CS.PA",
    "EL.PA", "AI.PA", "SAF.PA", "BN.PA", "DG.PA", "KER.PA", "ACA.PA", "ENGI.PA",
    "STLA.PA", "SU.PA", "GLE.PA", "VIV.PA", "ML.PA", "PUB.PA", "CAP.PA", "RNO.PA",
    "RI.PA", "ORA.PA", "WLN.PA", "VIE.PA", "BVI.PA", "ATO.PA", "EN.PA", "URW.PA",
    "EDEN.PA", "DSY.PA", "AKE.PA", "LR.PA", "FR.PA", "ALO.PA", "HO.PA", "CA.PA",
    "POM.PA", "GTT.PA", "FDJ.PA", "SCR.PA", "NK.PA", "TEP.PA",

    # === ALLEMAGNE (.DE) ===
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "AIR.DE", "MUV2.DE", "MBG.DE", "BAS.DE",
    "BMW.DE", "BAYN.DE", "VOW3.DE", "ADS.DE", "DPW.DE", "DBK.DE", "P911.DE",
    "RWE.DE", "EOAN.DE", "IFX.DE", "BEI.DE", "HEI.DE", "MRK.DE", "HEN3.DE",
    "VNA.DE", "FRE.DE", "CON.DE", "PAH3.DE", "SY1.DE", "SHL.DE", "PUM.DE", "MTX.DE",
    "DB1.DE", "ZAL.DE", "1COV.DE", "QIA.DE", "BNR.DE", "FME.DE", "EVK.DE", "LHA.DE",
    "TKA.DE", "DTG.DE", "SRT3.DE", "HNR1.DE", "RHM.DE", "G1A.DE", "BOSS.DE",
    "LIN.DE", "NEM.DE", "KGX.DE", "AFX.DE",

    # === SUISSE (.SW) ===
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ABBN.SW", "ZURN.SW", "CFR.SW",
    "GIVN.SW", "HOLN.SW", "SIKA.SW", "SLHN.SW", "LONN.SW", "GEBN.SW", "PGHN.SW",
    "SCMN.SW", "SOON.SW", "SREN.SW", "ALC.SW", "UHR.SW", "ADEN.SW", "BAER.SW",
    "STMN.SW", "KNIN.SW", "TEMN.SW", "EMSN.SW", "VACN.SW", "GALD.SW",

    # === PAYS-BAS (.AS) ===
    "ASML.AS", "PRX.AS", "AD.AS", "INGA.AS", "WKL.AS", "PHIA.AS", "HEIA.AS", "DSM.AS",
    "ABN.AS", "REN.AS", "ASRNL.AS", "AKZA.AS", "RAND.AS", "KPN.AS", "NN.AS",
    "IMCD.AS", "AGN.AS", "MT.AS", "URW.AS", "BESI.AS", "JDEP.AS",

    # === ESPAGNE (.MC) ===
    "ITX.MC", "IBE.MC", "SAN.MC", "BBVA.MC", "TEF.MC", "REP.MC", "FER.MC", "AENA.MC",
    "AMS.MC", "ELE.MC", "CABK.MC", "ACS.MC", "GRF.MC", "MAP.MC", "RED.MC", "ENG.MC",
    "BKT.MC", "SAB.MC", "MEL.MC", "ANA.MC", "CLNX.MC", "NTGY.MC",

    # === ITALIE (.MI) ===
    "ENI.MI", "ISP.MI", "STLAM.MI", "G.MI", "UCG.MI", "ENEL.MI", "RACE.MI", "MB.MI",
    "PRY.MI", "TIT.MI", "ATL.MI", "CNHI.MI", "BAMI.MI", "NEXI.MI", "AMP.MI",
    "MONC.MI", "BMED.MI", "REC.MI", "CPR.MI", "DIA.MI", "SPM.MI", "TEN.MI",
    "FCT.MI", "INW.MI", "LDO.MI",

    # === SUÈDE (.ST) ===
    "VOLV-B.ST", "ATCO-A.ST", "INVE-B.ST", "ASSA-B.ST", "ERIC-B.ST", "HEXA-B.ST",
    "EVO.ST", "SAND.ST", "ESSITY-B.ST", "SEB-A.ST", "SHB-A.ST", "SKF-B.ST",
    "TEL2-B.ST", "GETI-B.ST", "ALFA.ST", "ELUX-B.ST", "BOL.ST", "SCA-B.ST",
    "SWED-A.ST", "TELIA.ST", "EQT.ST", "NDA-SE.ST",

    # === DANEMARK (.CO) ===
    "NOVO-B.CO", "DSV.CO", "ORSTED.CO", "MAERSK-B.CO", "CARL-B.CO", "DANSKE.CO",
    "GMAB.CO", "PNDORA.CO", "VWS.CO", "COLO-B.CO", "TRYG.CO", "GN.CO", "ROCK-B.CO",
    "AMBU-B.CO", "DEMANT.CO",

    # === FINLANDE (.HE) ===
    "NOKIA.HE", "KNEBV.HE", "NESTE.HE", "SAMPO.HE", "FORTUM.HE", "UPM.HE",
    "OUT1V.HE", "STERV.HE", "ELISA.HE", "WRT1V.HE", "NDA-FI.HE", "TYRES.HE",

    # === NORVÈGE (.OL) ===
    "EQNR.OL", "DNB.OL", "TEL.OL", "MOWI.OL", "AKERBP.OL", "NHY.OL", "ORK.OL",
    "YAR.OL", "STB.OL", "SUBC.OL", "SCHA.OL", "GJF.OL",

    # === BELGIQUE (.BR) ===
    "ABI.BR", "KBC.BR", "UCB.BR", "AGS.BR", "SOLB.BR", "ARGX.BR", "PROX.BR",
    "GBLB.BR", "UMI.BR", "COFB.BR", "ELI.BR",

    # === IRLANDE (.IR) ===
    "KRZ.IR", "KSP.IR", "RY4C.IR",
]

def get_universe():
    """Retourne la liste dédupliquée des tickers."""
    return list(dict.fromkeys(STOXX_600_TICKERS))

if __name__ == "__main__":
    tickers = get_universe()
    print(f"Univers : {len(tickers)} tickers")
    # Répartition par marché
    from collections import Counter
    suffixes = [t.split(".")[-1] for t in tickers]
    print(Counter(suffixes))
