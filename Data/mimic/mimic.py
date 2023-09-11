import pandas as pd

def get_all_icustay_ids(engine, heart_only=False):
    with engine.connect() as connection:
        if heart_only:
            sql = "select distinct icustay_id from icustays WHERE first_careunit IN ('CSRU', 'CCU') AND last_careunit IN ('CSRU', 'CCU')"
        else:
            sql = "select distinct icustay_id from icustays"
        stays = list(pd.read_sql(sql, connection)['icustay_id'])
    return stays

# def get_events(connection, chart_ids, icustay_id):
#
#
#     events = f"SELECT hadm_id, itemid, charttime, valuenum, valueuom " \
#              f"FROM (SELECT hadm_id, itemid, charttime, valuenum, valueuom " \
#              f"FROM chartevents WHERE itemid in {chart_ids} AND hadm_id IN (SELECT hadm_id FROM icustays " \
#              f"WHERE first_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') " \
#              f"AND last_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') and icustay_id in {icustay_id})) as events "
#
#
#
#     sql2 = f"SELECT DISTINCT icu.icustay_id, e.hadm_id, e.charttime " \
#            f"FROM icustays icu, ({events}) as e " \
#            f"WHERE e.hadm_id = icu.hadm_id " \
#            f"AND e.charttime BETWEEN icu.intime AND icu.outtime " \
#
#
#
#     sql3 = f"SELECT e.hadm_id, t.icustay_id, e.itemid, e.charttime, " \
#            f"e.valuenum " \
#            f"FROM ({events}) as e LEFT JOIN ({sql2}) as t " \
#            f"ON e.hadm_id = t.hadm_id AND e.charttime = t.charttime " \
#            f"ORDER BY e.hadm_id, t.icustay_id, e.charttime"
#
#     events = pd.read_sql(sql3, con=connection)
#     # events.drop_duplicates(inplace=True)
#
#     return events

def get_events(connection, chart_ids, lab_ids, icustay_ids):


    # events = f"SELECT hadm_id, itemid, charttime, valuenum, valueuom " \
    #          f"FROM (SELECT hadm_id, itemid, charttime, valuenum, valueuom " \
    #          f"FROM chartevents WHERE itemid in {chart_ids} AND hadm_id IN (SELECT hadm_id FROM icustays " \
    #          f"WHERE first_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') " \
    #          f"AND last_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') and icustay_id in {tuple(icustay_ids)}))" \
    #          f"UNION " \
    #          f"SELECT hadm_id, itemid,charttime,valuenum, valueuom " \
    #          f"FROM labevents " \
    #          f"WHERE itemid in {tuple(lab_ids)} " \
    #          f"AND hadm_id IN (SELECT hadm_id FROM icustays " \
    #          f"WHERE first_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') " \
    #          f"AND last_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU')) as events " \
    #          f"ORDER BY hadm_id, charttime"
    events = f"SELECT hadm_id, itemid, charttime, valuenum, valueuom " \
             f"FROM (SELECT hadm_id, itemid, charttime, valuenum, valueuom " \
             f"FROM chartevents WHERE itemid in {chart_ids} AND hadm_id IN (SELECT hadm_id FROM icustays " \
             f"WHERE first_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') " \
             f"AND last_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') and icustay_id in {tuple(icustay_ids)}) " \
             f"UNION " \
             f"SELECT hadm_id, itemid,charttime,valuenum, valueuom " \
             f"FROM labevents " \
             f"WHERE itemid in {lab_ids} " \
             f"AND hadm_id IN (SELECT hadm_id FROM icustays " \
             f"WHERE first_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') " \
             f"AND last_careunit IN ('CSRU', 'CCU', 'MICU', 'SICU', 'TSICU') and icustay_id in {tuple(icustay_ids)}) ) as events " \
             f"ORDER BY hadm_id, charttime"



    sql2 = f"SELECT DISTINCT icu.icustay_id, e.hadm_id, e.charttime " \
           f"FROM icustays icu, ({events}) as e " \
           f"WHERE e.hadm_id = icu.hadm_id " \
           f"AND e.charttime BETWEEN icu.intime AND icu.outtime " \



    sql3 = f"SELECT e.hadm_id, t.icustay_id, e.itemid, e.charttime, " \
           f"e.valuenum " \
           f"FROM ({events}) as e LEFT JOIN ({sql2}) as t " \
           f"ON e.hadm_id = t.hadm_id AND e.charttime = t.charttime  WHERE t.icustay_id in {tuple(icustay_ids)}" \
           f"ORDER BY e.hadm_id, t.icustay_id, e.charttime"

    events = pd.read_sql(sql3, con=connection)
    # events.drop_duplicates(inplace=True)

    return events