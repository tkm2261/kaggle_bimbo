# encoding: utf-8

import pandas
import sqlite3
con = sqlite3.connect("data.db")
# -*- coding: utf-8 -*-
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import sessionmaker


SERVER = "192.168.99.68"
PORT = 5432
DB = "pj020102"
SCHEMA = "a_vehicle"
USER = "takami.sato"
PASSWD = "!4et$GJ8"


SQL_CONNECTION_URL_FORMAT = 'postgresql://{user}:{passwd}@{server}:{port}/{db}'

_sql_connection_url = \
                SQL_CONNECTION_URL_FORMAT.format(user=USER,
                                                   passwd=PASSWD,
                                                   server=SERVER,
                                                   port=PORT,
                                                   db=DB)

engine = create_engine(_sql_connection_url, echo=False)

if __name__ == '__main__':
    con.execute('DROP TABLE IF EXISTS train;')
    con.commit()
    cnt = 0
    for chunk in pandas.read_csv('train.csv.gz', chunksize=10000, compression='gzip'):
        cnt += 1
        print 'chunk:', cnt
        chunk.to_sql('train', engine, if_exists='append', index=False)
    exit()

    pandas.read_csv('product_master.csv', index_col='Producto_ID').to_sql('product_master', con, if_exists='replace')
    pandas.read_csv('agencia_master.csv', index_col='Agencia_ID').to_sql('agencia_master', con, if_exists='replace')
    
    
    con.execute('DROP TABLE IF EXISTS agencia_trans;')
    con.commit()
    con.execute("""
    CREATE TABLE agencia_trans AS
    SELECT
        Agencia_ID,
        count(*) as ag_cnt,
        max(Venta_uni_hoy) ag_max_venta_uni,
        min(CASE WHEN Venta_uni_hoy > 0 THEN Venta_uni_hoy ELSE NULL END) ag_min_venta_uni,
        avg(Venta_uni_hoy) ag_avg_venta_uni,
        
        max(Venta_hoy) ag_max_venta_hoy,
        min(CASE WHEN Venta_hoy > 0 THEN Venta_hoy ELSE NULL END) ag_min_venta_hoy,
        avg(Venta_hoy) ag_avg_venta_hoy,
        
        max(Dev_uni_proxima) ag_max_dev_uni,
        min(CASE WHEN Dev_uni_proxima > 0 THEN Dev_uni_proxima ELSE NULL END) ag_min_dev_uni,
        avg(Dev_uni_proxima) ag_avg_dev_uni,
        
        max(Dev_proxima) ag_max_dev_hoy,
        min(CASE WHEN Dev_proxima > 0 THEN Dev_proxima ELSE NULL END) ag_min_dev_hoy,
        avg(Dev_proxima) ag_avg_dev_hoy,
        
        max(Demanda_uni_equil) ag_max_demand,
        min(CASE WHEN Demanda_uni_equil > 0 THEN Demanda_uni_equil ELSE NULL END) ag_min_demand,
        avg(Demanda_uni_equil) ag_avg_demand
    FROM
        train
    GROUP BY
        Agencia_ID;
    """)
    con.commit()
    
    
    con.execute('DROP TABLE IF EXISTS canal_trans;')
    con.commit()
    con.execute("""
    CREATE TABLE canal_trans AS
    SELECT
        Canal_ID,
        count(*) as ch_cnt,
        max(Venta_uni_hoy) ch_max_venta_uni,
        min(CASE WHEN Venta_uni_hoy > 0 THEN Venta_uni_hoy ELSE NULL END) ch_min_venta_uni,
        avg(Venta_uni_hoy) ch_avg_venta_uni,
        
        max(Venta_hoy) ch_max_venta_hoy,
        min(CASE WHEN Venta_hoy > 0 THEN Venta_hoy ELSE NULL END) ch_min_venta_hoy,
        avg(Venta_hoy) ch_avg_venta_hoy,
        
        max(Dev_uni_proxima) ch_max_dev_uni,
        min(CASE WHEN Dev_uni_proxima > 0 THEN Dev_uni_proxima ELSE NULL END) ch_min_dev_uni,
        avg(Dev_uni_proxima) ch_avg_dev_uni,
        
        max(Dev_proxima) ch_max_dev_hoy,
        min(CASE WHEN Dev_proxima > 0 THEN Dev_proxima ELSE NULL END) ch_min_dev_hoy,
        avg(Dev_proxima) ch_avg_dev_hoy,
        
        max(Demanda_uni_equil) ch_max_demand,
        min(CASE WHEN Demanda_uni_equil > 0 THEN Demanda_uni_equil ELSE NULL END) ch_min_demand,
        avg(Demanda_uni_equil) ch_avg_demand
    FROM
        train
    GROUP BY
        Canal_ID;
    """)
    con.commit()
    
    con.execute('DROP TABLE IF EXISTS cliente_trans;')
    con.commit()
    con.execute("""
    CREATE TABLE cliente_trans AS
    SELECT
        Cliente_ID,
        count(*) as cl_cnt,
        max(Venta_uni_hoy) cl_max_venta_uni,
        min(CASE WHEN Venta_uni_hoy > 0 THEN Venta_uni_hoy ELSE NULL END) cl_min_venta_uni,
        avg(Venta_uni_hoy) cl_avg_venta_uni,
        
        max(Venta_hoy) cl_max_venta_hoy,
        min(CASE WHEN Venta_hoy > 0 THEN Venta_hoy ELSE NULL END) cl_min_venta_hoy,
        avg(Venta_hoy) cl_avg_venta_hoy,
        
        max(Dev_uni_proxima) cl_max_dev_uni,
        min(CASE WHEN Dev_uni_proxima > 0 THEN Dev_uni_proxima ELSE NULL END) cl_min_dev_uni,
        avg(Dev_uni_proxima) cl_avg_dev_uni,
        
        max(Dev_proxima) cl_max_dev_hoy,
        min(CASE WHEN Dev_proxima > 0 THEN Dev_proxima ELSE NULL END) cl_min_dev_hoy,
        avg(Dev_proxima) cl_avg_dev_hoy,
        
        max(Demanda_uni_equil) cl_max_demand,
        min(CASE WHEN Demanda_uni_equil > 0 THEN Demanda_uni_equil ELSE NULL END) cl_min_demand,
        avg(Demanda_uni_equil) cl_avg_demand
    FROM
        train
    GROUP BY
        Cliente_ID;
    """)
    con.commit()