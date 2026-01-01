# def filter_by_market(results, market):
#     """
#     Filter stocks by market. Currently placeholder: returns all.
#     You can extend this using info['exchange'].
#     """
#     # TODO: Implement proper filtering based on exchange
#     return results

# def filter_by_market(results, market):
#     filtered = []

#     for r in results:
#         symbol = r["symbol"]

#         if market == "NSE" and symbol.endswith(".NS"):
#             filtered.append(r)
#         elif market == "BSE" and symbol.endswith(".BO"):
#             filtered.append(r)
#         elif market == "NYSE" and not symbol.endswith((".NS", ".BO")):
#             filtered.append(r)

#     return filtered

def filter_by_market(results, market):
    filtered = []

    for r in results:
        symbol = r["symbol"].upper()

        if market == "NSE" and symbol.endswith(".NS"):
            filtered.append(r)
        elif market == "BSE" and symbol.endswith(".BO"):
            filtered.append(r)
        elif market == "NYSE" and not symbol.endswith((".NS", ".BO")):
            filtered.append(r)

    return filtered


