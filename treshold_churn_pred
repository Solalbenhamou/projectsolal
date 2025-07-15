#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery
from pytz import timezone

def fetch_predictions(client):
    """
    Récupère toutes les prédictions de churn depuis BigQuery.
    """
    sql = """
    SELECT
      shop_id,
      run_date,
      proba_churn,
      group_number
    FROM
      `warehouse-459819.Predictions.prediction_group_enriched`
    """
    return client.query(sql).to_dataframe()

def fetch_shop_ids(client, shop_name):
    """
    Récupère les shop_id correspondant au nom de l'entreprise (insensible à la casse).
    """
    sql = """
    SELECT shop_id
    FROM `warehouse-459819.analytics_test.stg_shop`
    WHERE LOWER(shop_name) = LOWER(@shop_name)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("shop_name", "STRING", shop_name)
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()

def process_shop(df_preds, shop_id, shop_name, threshold, output_dir):
    """
    Filtre les données pour un shop donné, trace et enregistre les résultats.
    """
    # Conversion de la date et timezone
    tz = timezone('Asia/Jerusalem')
    df = df_preds.copy()
    df['run_date'] = pd.to_datetime(df['run_date']).dt.tz_localize(tz, ambiguous='NaT')
    today = pd.Timestamp.now(tz).normalize()

    # Filtrer sur la journée courante
    mask = (
        (df['shop_id'] == shop_id) &
        (df['run_date'] >= today) &
        (df['run_date'] < today + pd.Timedelta(days=1))
    )
    df_today = df.loc[mask]

    # Compter les proba_churn > seuil par group_number
    counts_over = (
        df_today
        .groupby('group_number')['proba_churn']
        .apply(lambda s: (s > threshold).sum())
        .loc[lambda s: s > 0]
    )
    if counts_over.empty:
        print(f"Aucun churn > {threshold*100:.0f}% pour {shop_name} (ID:{shop_id})")
        return

    # Tracé et enregistrement
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts_over.index.astype(str), counts_over.values)
    ax.set(
        xlabel='Group Number',
        ylabel=f"Clients (proba_churn > {threshold*100:.0f}%)",
        title=f"{shop_name} (ID:{shop_id}) — churn > {threshold*100:.0f}%"
    )
    fig.tight_layout()

    # Sauvegarde
    png_path = os.path.join(output_dir, f"{shop_name}_{shop_id}_churn.png")
    csv_path = os.path.join(output_dir, f"{shop_name}_{shop_id}_churn_counts.csv")
    fig.savefig(png_path)
    counts_over.to_csv(csv_path, header=['count'])
    plt.close(fig)

    print(f"Résultats enregistrés : {png_path}, {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyse quotidienne du churn par groupe pour un shop donné"
    )
    parser.add_argument(
        '--shop_name', required=True,
        help="Nom de l'entreprise (insensible à la casse)"
    )
    parser.add_argument(
        '--threshold_pct', type=float, required=True,
        help="Seuil de churn en pourcentage (ex: 80 pour 80%)"
    )
    parser.add_argument(
        '--output_dir', default='outputs',
        help="Dossier de destination pour les fichiers générés"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Le client BigQuery utilisera GOOGLE_APPLICATION_CREDENTIALS et GCP_PROJECT
    project = os.getenv('GCP_PROJECT')
    client = bigquery.Client(project=project)

    df_preds = fetch_predictions(client)
    df_shops = fetch_shop_ids(client, args.shop_name)
    if df_shops.empty:
        print(f"Aucun shop_id trouvé pour '{args.shop_name}'")
        return

    threshold = args.threshold_pct / 100
    for sid in df_shops['shop_id']:
        process_shop(df_preds, sid, args.shop_name, threshold, args.output_dir)

if __name__ == '__main__':
    main()
