import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import networkx as nx
import math
import itertools
from collections import Counter
import re
import string

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14

HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)


class HealthfluencerForensics:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.clean_and_enrich_data()

    def _extract_hashtags(self, text: str):
        if pd.isna(text) or text is None:
            return None
        matches = HASHTAG_RE.findall(str(text))
        # normalize: strip punctuation around match and lowercase; ignore empty results
        cleaned = [
            m.strip().lower().strip(string.punctuation)
            for m in matches
            if m and m.strip().strip(string.punctuation)
        ]

        return cleaned if cleaned else None

    def clean_and_enrich_data(self):
        if "create_time" in self.df.columns:
            try:
                self.df["timestamp"] = pd.to_datetime(self.df["create_time"], unit="s")
            except:
                self.df["timestamp"] = pd.to_datetime(self.df["create_time"])

            self.df["hour_of_day"] = self.df["timestamp"].dt.hour
            self.df["day_of_week"] = self.df["timestamp"].dt.day_name()
            self.df["month_period"] = self.df["timestamp"].dt.to_period("M")

        # metric normalization (fill nans with 0 for count data)
        metrics = ["play_count", "digg_count", "comment_count", "share_count"]
        for m in metrics:
            if m in self.df.columns:
                self.df[m] = self.df[m].fillna(0)

        if "desc" in self.df.columns:
            self.df["extracted_hashtags"] = self.df["desc"].apply(
                self._extract_hashtags
            )

        # calculate engagement rate by view
        # (likes + comments + shares) / views * 100
        self.df["total_interactions"] = (
            self.df["digg_count"] + self.df["comment_count"] + self.df["share_count"]
        )
        self.df["engagement_rate"] = self.df.apply(
            lambda x: (
                (x["total_interactions"] / x["play_count"] * 100)
                if x["play_count"] > 0
                else 0
            ),
            axis=1,
        )

        # calculate amplification and conversation ratios
        self.df["amplification_ratio"] = self.df.apply(
            lambda x: x["share_count"] / x["digg_count"] if x["digg_count"] > 0 else 0,
            axis=1,
        )
        self.df["conversation_ratio"] = self.df.apply(
            lambda x: (
                x["comment_count"] / x["digg_count"] if x["digg_count"] > 0 else 0
            ),
            axis=1,
        )

    def benfords_law_analysis(self, metric="play_count"):
        """
        Analyze the leading digit distribution of a metric against Benford's Law.
        Violation suggests artificial manipulation e.g., botting.
        """
        if metric not in self.df.columns:
            return None

        data = pd.to_numeric(self.df[metric], errors="coerce").dropna().astype(str)
        leading_digits = data.str[0].astype(int)
        observed_counts = leading_digits.value_counts().sort_index()

        for d in range(1, 10):
            if d not in observed_counts:
                observed_counts[d] = 0
        observed_counts = observed_counts.sort_index()

        # calculate expected counts (benford's law: p(d) = log10(1 + 1/d))
        total_obs = observed_counts.sum()
        expected_probs = [math.log10(1 + 1 / d) for d in range(1, 10)]
        expected_counts = np.array(expected_probs) * total_obs

        observed_probs = observed_counts.to_numpy() / total_obs

        # chi-square test (sensitive to n)
        chi2_stat, p_val = stats.chisquare(
            f_obs=observed_counts.values, f_exp=expected_counts
        )

        # mean absolute deviation (mad) - robust to n
        mad = np.mean(np.abs(observed_probs - expected_probs))

        # mad thresholds (nigrini)
        if mad <= 0.006:
            mad_verdict = "Close Conformity"
        elif mad <= 0.012:
            mad_verdict = "Acceptable Conformity"
        elif mad <= 0.015:
            mad_verdict = "Marginally Acceptable"
        else:
            mad_verdict = "Nonconformity"

        return {
            "observed": observed_counts,
            "expected": expected_counts,
            "chi2": chi2_stat,
            "p_value": p_val,
            "mad": mad,
            "mad_verdict": mad_verdict,
            # chi-square verdict is strictly mathematical, mad is forensic
            "chi_verdict": "Suspicious" if p_val < 0.05 else "Consistent",
        }

    def detect_bot_anomalies(self):
        """
        Use Isolation Forest to identify multivariate anomalies.
        """
        features = [
            "play_count",
            "digg_count",
            "comment_count",
            "share_count",
            "engagement_rate",
        ]
        valid_features = [f for f in features if f in self.df.columns]
        X = self.df[valid_features].dropna()

        if X.empty:
            return None

        # standardize features (mean=0, std=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # isolation forest
        # contamination=0.05 assumes ~5% of data is anomalous
        clf = IsolationForest(contamination=0.05, random_state=42)
        self.df["anomaly_score"] = clf.fit_predict(X_scaled)
        # -1 indicates anomaly, 1 indicates normal
        return self.df[self.df["anomaly_score"] == -1]

    def analyze_hashtag_network(self, top_n=50):
        """
        Construct a graph of hashtag co-occurrences to identify content clusters.
        High modularity suggests distinct echo chambers e.g #antivax vs #science.
        """
        G = nx.Graph()

        hashtag_lists = self.df["extracted_hashtags"].dropna()

        # build edges
        for tags in hashtag_lists:
            if len(tags) > 1:
                # add edges between all pairs in the list
                for t1, t2 in itertools.combinations(tags, 2):
                    if G.has_edge(t1, t2):
                        G[t1][t2]["weight"] += 1
                    else:
                        G.add_edge(t1, t2, weight=1)

        if len(G.nodes) == 0:
            print("Warning: No hashtag connections found.")
            return None, None

        # prune graph: keep only significant edges and top nodes for visualization
        # calculate degree centrality
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality, key=lambda n: centrality[n], reverse=True)[
            :top_n
        ]
        G_sub = G.subgraph(top_nodes)

        return G_sub, centrality

    def analyze_monthly_hashtag_trends(self, top_n_per_month=5):
        """
        Analyzes the top N hashtags for each month
        Returns a Pivot Table for heatmap visualization
        """
        if (
            "extracted_hashtags" not in self.df.columns
            or "month_period" not in self.df.columns
        ):
            print("Required columns for monthly analysis missing.")
            return None

        # explode the hashtags so each tag has its own row with a timestamp
        exploded_df = self.df.explode("extracted_hashtags").dropna(
            subset=["extracted_hashtags"]
        )

        # group by month and hashtag, then count
        monthly_counts = (
            exploded_df.groupby(["month_period", "extracted_hashtags"], dropna=False)
            .size()
            .reset_index(name="count")
        )

        print(f"\n{'Month':<10} | {'Top Hashtags (Count)'}")
        print("-" * 80)
        for period, group in monthly_counts.groupby("month_period"):
            top_tags = group.nlargest(top_n_per_month, "count")
            formatted_tags = [
                f"{row['extracted_hashtags']}({row['count']})"
                for _, row in top_tags.iterrows()
            ]
            tags_str = ", ".join(formatted_tags)
            print(f"{str(period):<10} | {tags_str}")

        # filter to get top n hashtags for each month
        top_monthly = (
            monthly_counts.sort_values(
                ["month_period", "count"], ascending=[True, False]
            )
            .groupby("month_period", group_keys=True)
            .head(top_n_per_month)
            .reset_index(drop=True)
        )

        unique_top_tags = top_monthly["extracted_hashtags"].unique()

        # filter the original counts to include only these significant tags across all months
        # allows us to see when a top tag fades in/out
        relevant_data = monthly_counts[
            monthly_counts["extracted_hashtags"].isin(unique_top_tags)
        ]

        # pivot for heatmap: index=hashtag, columns=month, values=count
        hashtag_pivot = relevant_data.pivot_table(
            index="extracted_hashtags",
            columns="month_period",
            values="count",
            fill_value=0,
        )

        return hashtag_pivot

    def generate_report_graphs(self, benford_res, anomalies, G_sub, hashtag_pivot):
        if benford_res:
            plt.figure()
            digits = range(1, 10)
            plt.bar(
                digits,
                benford_res["observed"],
                alpha=0.6,
                label="Observed",
                color="steelblue",
            )
            plt.plot(
                digits,
                benford_res["expected"],
                "r--o",
                linewidth=2,
                label="Benford's Law",
            )
            title = (
                f"Forensic Analysis of View Counts\n"
                f"Chi-Sq p={benford_res['p_value']:.4f} ({benford_res['chi_verdict']})\n"
                f"MAD={benford_res['mad']:.4f} ({benford_res['mad_verdict']})"
            )
            plt.title(title)
            plt.xlabel("Leading Digit")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig("benford.png")
            # plt.show()

        if anomalies is not None and not anomalies.empty:
            plt.figure()
            normal = self.df[self.df["anomaly_score"] == 1]
            outliers = self.df[self.df["anomaly_score"] == -1]

            total_count = len(self.df)
            anomaly_count = len(outliers)
            anomaly_pct = (anomaly_count / total_count) * 100

            plt.scatter(
                normal["play_count"],
                normal["engagement_rate"],
                c="grey",
                alpha=0.3,
                label="Organic",
            )
            plt.scatter(
                outliers["play_count"],
                outliers["engagement_rate"],
                c="red",
                marker="x",
                label="Anomalous (Bot/Viral)",
            )

            plt.xscale("log")  # Log scale for Power Law data
            plt.title("Isolation Forest Anomaly Detection: Views vs Engagement")
            plt.title(
                f"Isolation Forest Anomaly Detection: Views vs Engagement\n{anomaly_count} of {total_count} ({anomaly_pct:.2f}%)"
            )
            plt.xlabel("Play Count (Log Scale)")
            plt.ylabel("Engagement Rate (%)")
            plt.legend()
            plt.savefig("anomaly.png")
            # plt.show()

        if G_sub and G_sub.number_of_nodes() > 0:
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G_sub, k=0.5, iterations=50)
            d = dict(G_sub.degree)

            nx.draw_networkx_nodes(
                G_sub,
                pos,
                node_size=[v * 50 for v in d.values()],
                node_color="purple",
                alpha=0.6,
            )
            nx.draw_networkx_edges(G_sub, pos, alpha=0.2)
            nx.draw_networkx_labels(G_sub, pos, font_size=10, font_weight="bold")

            plt.title(f"Hashtag Co-occurrence Topology (Top {len(G_sub.nodes)} Nodes)")
            plt.axis("off")
            plt.savefig("netgraph.png")
            # plt.show()

        if hashtag_pivot is not None and not hashtag_pivot.empty:
            n_rows, n_cols = hashtag_pivot.shape
            figsize = (max(12, n_cols * 1.2), max(6, n_rows * 0.7))
            fig, ax = plt.subplots(figsize=figsize)

            sns.heatmap(
                hashtag_pivot,
                cmap="YlGnBu",
                annot=True,
                fmt="g",
                linewidths=0.5,
                annot_kws={"size": 20},
                cbar_kws={"shrink": 0.6},
                ax=ax,
            )

            y_positions = np.arange(n_rows) + 0.5
            ax.yaxis.set_major_locator(mticker.FixedLocator(y_positions))
            ax.yaxis.set_major_formatter(
                mticker.FixedFormatter(hashtag_pivot.index.tolist())
            )
            ax.set_title(
                "Evolution of Top Hashtags per Month (Frequency Heatmap)", fontsize=24
            )
            ax.set_xlabel("Month", fontsize=24)
            ax.set_ylabel("Hashtag", fontsize=24)
            ax.tick_params(axis="x", labelsize=18, rotation=45)
            ax.tick_params(axis="y", labelsize=18)
            plt.tight_layout()
            plt.savefig("hashtagevo.png", bbox_inches="tight", dpi=150)
            # plt.show()


top_n = 50
# analyst = HealthfluencerForensics("tiktok_medical_data_cleaned.csv")
analyst = HealthfluencerForensics("tt.csv")
benford_results = analyst.benfords_law_analysis()
flagged_accounts = analyst.detect_bot_anomalies()
graph, centrality = analyst.analyze_hashtag_network(top_n)
trend_pivot = analyst.analyze_monthly_hashtag_trends(top_n_per_month=5)
analyst.generate_report_graphs(benford_results, flagged_accounts, graph, trend_pivot)
