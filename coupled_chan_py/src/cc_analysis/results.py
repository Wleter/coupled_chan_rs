from .utility import CFG
import pandas as pd
import numpy as np
import numpy.typing as npt

LEVELS_COLUMNS = {"parameter", "levels"}
class LevelsData:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def read_json(filename: str) -> 'LevelsData':
        df = pd.read_json(CFG.DATA_PATH / filename, lines = True, precise_float=True)

        if set(df.columns) != LEVELS_COLUMNS:
            raise ValueError(f"Expected columns {LEVELS_COLUMNS}, got {set(df.columns)}")

        df = df.sort_values(by="parameter").reset_index(drop=True)

        return LevelsData(df)
    
    def parameters(self) -> npt.NDArray:
        return np.array(self.df["parameter"])

    def levels(self) -> npt.NDArray:
        return np.array(list(self.df["levels"]))

SMATRIX_COLUMNS = {
    "parameter", 
    "s_length_re",
    "s_length_im",
    "elastic_cross_section",
    "tot_inelastic_cross_section",
    "inelastic_cross_sections"
}
class SMatrixData:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def read_json(filename: str) -> 'SMatrixData':
        df = pd.read_json(CFG.DATA_PATH / filename, lines = True, precise_float=True)

        if set(df.columns) != SMATRIX_COLUMNS:
            raise ValueError(f"Expected columns {SMATRIX_COLUMNS}, got {set(df.columns)}")

        df = df.sort_values(by="parameter").reset_index(drop=True)

        return SMatrixData(df)
    
    def parameters(self) -> npt.NDArray:
        return np.array(self.df["parameter"])

    def s_length_re(self) -> npt.NDArray:
        return np.array(self.df["s_length_re"])
    
    def s_length_im(self) -> npt.NDArray:
        return np.array(self.df["s_length_im"])
    
    def elastic_cross_sect(self) -> npt.NDArray:
        return np.array(self.df["elastic_cross_section"])

    def tot_inalstic_cross_sect(self) -> npt.NDArray:
        return np.array(self.df["tot_inelastic_cross_section"])

    def inalstic_cross_sects(self) -> npt.NDArray:
        return np.array(list(self.df["inelastic_cross_sections"]))
    
BOUND_STATE_DATA = {
    "parameter", 
    "bound_parameter",
    "nodes",
    "occupations"
}
class BoundStateData:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def read_json(filename: str) -> 'BoundStateData':
        df = pd.read_json(CFG.DATA_PATH / filename, lines = True, precise_float=True)

        if set(df.columns) != BOUND_STATE_DATA:
            raise ValueError(f"Expected columns {BOUND_STATE_DATA}, got {set(df.columns)}")

        df = df.sort_values(by="parameter").reset_index(drop=True)

        return BoundStateData(df)
    
    def exchange_parameters(self):
        self.df["parameter"], self.df["bound_parameter"] = self.df["bound_parameter"], self.df["parameter"]
        self.df = self.df.sort_values(by="parameter").reset_index(drop=True)
    
    def parameters(self) -> npt.NDArray:
        return np.array(self.df["parameter"])

    def bound_parameters(self) -> npt.NDArray:
        return np.array(self.df["bound_parameter"])
    
    def nodes(self) -> npt.NDArray:
        return np.array(self.df["nodes"])
    
    def occupations(self) -> npt.NDArray:
        return np.array(list(self.df["occupations"]))
    
    def __iter__(self):
        return map(lambda node: BoundStateData(self.df[self.nodes() == node]), self.df["nodes"].unique())