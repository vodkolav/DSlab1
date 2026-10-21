from Benchmarking.utils import read_json
from sympy.parsing.latex import parse_latex
from sympy import pi, Float, lambdify

import pandas as pd
import io
import re


def parse_latex_with_fallback(s, strict=False, ):
    if isinstance(s,(int,float)):
        return Float(s)
    try:
          return parse_latex(s, strict, backend="lark")
    except :
        try:
            return parse_latex(s, strict, backend="antlr")
        except Exception as e:
            print(e)
            return 'ERROR'

def parse_md(nodetext):
    # val = val.replace("$","")
    # print(row.Param)

    check = re.findall(r"^(\#+.*?)$", nodetext ,  flags = re.M)
    if any(check):
       return None

    consts = nodetext.replace("$","")\
                     .replace("<br>","")\
                     .replace(r"C_{F0}", r"C_{FO}")\
                     .replace(r"C^*", r"C_v") 

    consts = re.sub(r" +\|", r"|", consts)
  
    # print(consts)
    stream = io.StringIO(consts)
    constsdf = pd.read_table(stream, sep="|", header=0, index_col=False, skipinitialspace=True,)\
      .dropna(axis=1, how='all')\
      .iloc[1:]
    constsdf.columns = [cn.strip() for cn in constsdf.columns]

    constsdf.dropna(how='all', inplace=True)

    field = "value" if "value" in constsdf.columns else "Formula"
    # print(constsdf)

    constsdf['expr'] = constsdf[field].apply(parse_latex_with_fallback)
    constsdf['Param'] = constsdf["Param"].apply(lambda x: parse_latex_with_fallback(x).name)

    return constsdf


class EquationsManager:

    def __init__(self, canvas):
        self.parse_canvas(canvas)


    def parse_canvas(self, pth):

        canvas = read_json(pth) 
        # node = canvas['nodes'][6]

        Formulas = [parse_md(node['text']) for node in canvas['nodes']]
        self.Formulas = pd.concat(Formulas)


    def make_replacements(self):
        wat = self.Formulas[["Param", "expr"]].set_index("Param")#.transpose()
        self.replacements = wat.to_dict(orient="dict")['expr']
        self.replacements['pi'] = pi


    def overrides(self, ovrds: dict):
        for k,v in ovrds.items():
            v = parse_latex_with_fallback(v)
            self.Formulas.loc[self.Formulas.Param == k,  "expr"] = v 


    def expression(self, name:str):
        return self.Formulas.loc[self.Formulas.Param == name,  "expr"].values[0]


    def substitute(self, expression):

        if expression.is_Equality:
            erhsfs = {sym.name for sym in expression.rhs.free_symbols}
        else: 
            erhsfs = {}

        if any(erhsfs):
            
            subst = {k:v  for k,v in self.replacements.items() if k in erhsfs}
            subst = {k:self.substitute(v) for k,v in subst.items()}
            subst
            expression = expression.subs(subst)
            expression = expression.rhs if expression.is_Equality else expression
        return expression.simplify()


    def lambdify(self, expression):

        vars = list(expression.free_symbols)

        l_func = lambdify(vars, expression, modules='numpy')

        return l_func