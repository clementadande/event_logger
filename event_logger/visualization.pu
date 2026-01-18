"""
EventVisualizer - Visualization layer for event logs
Creates charts and graphs for data science and process mining insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib/seaborn not installed. Install with: pip install event_logger[viz]")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class EventVisualizer:
    """
    Creates visualizations for event logs and insights.
    Supports both static (matplotlib/seaborn) and interactive (plotly) charts.
    """
    
    def __init__(self, insight_processor, style='seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer with an InsightProcessor instance.
        
        Args:
            insight_processor: InsightProcessor instance with loaded data
            style: matplotlib style (default: 'seaborn-v0_8-darkgrid')
        """
        self.ip = insight_processor
        self.ep = insight_processor.ep
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style)
            sns.set_palette("husl")
        
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def _check_matplotlib(self):
        """Check if matplotlib is available."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib/seaborn required. Install with: pip install event_logger[viz]")
    
    def _check_plotly(self):
        """Check if plotly is available."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required. Install with: pip install event_logger[viz]")
    
    # ========== DATA SCIENCE VISUALIZATIONS ==========
    
    def plot_session_overview(self, save=True, show=True):
        """DS Overview: Session counts, events, traces, tokens."""
        self._check_matplotlib()
        
        summary = self.ip.ds1_semantic_summary()
        counts = self.ip.ds2_counts()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Event Log Overview', fontsize=16, fontweight='bold')
        
        # 1. Basic counts
        categories = ['Sessions', 'Traces', 'Events', 'Tool Calls']
        values = [counts['sessions'], counts['traces'], counts['events'], counts['tool_calls']]
        colors = sns.color_palette("husl", 4)
        
        axes[0, 0].bar(categories, values, color=colors)
        axes[0, 0].set_title('Overall Counts', fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(values):
            axes[0, 0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 2. Token distribution
        token_dist = self.ip.ds7_token_distribution()
        token_types = ['Prompt', 'Completion', 'Thoughts']
        token_totals = [
            token_dist['prompt_tokens']['total'],
            token_dist['completion_tokens']['total'],
            token_dist['thoughts_tokens']['total']
        ]
        
        axes[0, 1].pie(token_totals, labels=token_types, autopct='%1.1f%%', colors=colors[:3])
        axes[0, 1].set_title('Token Distribution', fontweight='bold')
        
        # 3. Events per trace distribution
        dist = self.ip.ds3_distribution_traces_events()
        events_data = dist['events_per_trace']
        
        data_text = f"""
        Mean: {events_data['mean']:.1f}
        Median: {events_data['median']:.1f}
        Min: {events_data['min']}
        Max: {events_data['max']}
        Std: {events_data['std']:.1f}
        """
        
        axes[1, 0].text(0.5, 0.5, data_text, ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 0].set_title('Events per Trace Statistics', fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Success vs Failure
        success_failure = self.ip.ds9_success_failure_rate()
        success_labels = ['Successful', 'Failed']
        success_values = [success_failure['successful_traces'], success_failure['failed_traces']]
        
        axes[1, 1].bar(success_labels, success_values, color=['green', 'red'])
        axes[1, 1].set_title('Trace Outcomes', fontweight='bold')
        axes[1, 1].set_ylabel('Count')
        for i, v in enumerate(success_values):
            axes[1, 1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'session_overview.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'session_overview.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_tool_usage(self, top_n=10, save=True, show=True):
        """DS6: Tool frequency distribution."""
        self._check_matplotlib()
        
        tool_freq = self.ip.ds6_tool_frequency()
        
        if not tool_freq:
            print("⚠️ No tool usage data available")
            return
        
        # Sort and get top N
        sorted_tools = sorted(tool_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        tools, counts = zip(*sorted_tools)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(tools))
        bars = ax.barh(range(len(tools)), counts, color=colors)
        
        ax.set_yticks(range(len(tools)))
        ax.set_yticklabels(tools)
        ax.set_xlabel('Call Count', fontweight='bold')
        ax.set_title(f'Top {len(tools)} Most Used Tools', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'tool_usage.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'tool_usage.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_token_consumption(self, save=True, show=True):
        """DS7-DS8: Token usage patterns."""
        self._check_matplotlib()
        
        token_dist = self.ip.ds7_token_distribution()
        token_consumption = self.ip.ds8_token_consumption()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Token Usage Analysis', fontsize=16, fontweight='bold')
        
        # 1. Token type breakdown
        token_types = ['Prompt', 'Completion', 'Thoughts']
        totals = [
            token_dist['prompt_tokens']['total'],
            token_dist['completion_tokens']['total'],
            token_dist['thoughts_tokens']['total']
        ]
        
        colors = ['#3498db', '#e74c3c', '#f39c12']
        axes[0].bar(token_types, totals, color=colors)
        axes[0].set_title('Total Tokens by Type', fontweight='bold')
        axes[0].set_ylabel('Token Count')
        
        for i, v in enumerate(totals):
            axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Per session vs per trace
        categories = ['Per Session (avg)', 'Per Trace (avg)']
        values = [
            token_consumption['tokens_per_session']['mean'],
            token_consumption['tokens_per_trace']['mean']
        ]
        
        axes[1].bar(categories, values, color=['#2ecc71', '#9b59b6'])
        axes[1].set_title('Average Token Consumption', fontweight='bold')
        axes[1].set_ylabel('Tokens')
        
        for i, v in enumerate(values):
            axes[1].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'token_consumption.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'token_consumption.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_duration_distributions(self, save=True, show=True):
        """DS4: Session and trace duration distributions."""
        self._check_matplotlib()
        
        durations = self.ip.ds4_duration_distributions()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Duration Distributions', fontsize=16, fontweight='bold')
        
        # Session durations
        session_stats = durations['session_duration_seconds']
        session_data = f"""
        Mean: {session_stats['mean']:.1f}s
        Median: {session_stats['median']:.1f}s
        Min: {session_stats['min']:.1f}s
        Max: {session_stats['max']:.1f}s
        """
        
        axes[0].text(0.5, 0.5, session_data, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        axes[0].set_title('Session Duration', fontweight='bold')
        axes[0].axis('off')
        
        # Trace durations
        trace_stats = durations['trace_duration_seconds']
        trace_data = f"""
        Mean: {trace_stats['mean']:.1f}s
        Median: {trace_stats['median']:.1f}s
        Min: {trace_stats['min']:.1f}s
        Max: {trace_stats['max']:.1f}s
        """
        
        axes[1].text(0.5, 0.5, trace_data, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        axes[1].set_title('Trace Duration', fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'duration_distributions.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'duration_distributions.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_correlation_heatmap(self, save=True, show=True):
        """DS10: Feature correlation with failure."""
        self._check_matplotlib()
        
        corr = self.ip.ds10_feature_correlation()
        
        if corr.empty:
            print("⚠️ Not enough data for correlation analysis")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create correlation matrix for all features
        corr_df = pd.DataFrame({
            'Feature': corr.index,
            'Correlation with Failure': corr.values
        }).set_index('Feature')
        
        sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation with Trace Failure', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'correlation_heatmap.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # ========== PROCESS MINING VISUALIZATIONS ==========
    
    def plot_process_flow(self, top_n=5, save=True, show=True):
        """PM4: Happy paths visualization."""
        self._check_matplotlib()
        
        happy_paths = self.ip.pm4_happy_paths()
        
        if not happy_paths:
            print("⚠️ No process paths available")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Display top N paths
        for i, path_info in enumerate(happy_paths[:top_n]):
            sequence = path_info['sequence']
            frequency = path_info['frequency']
            percentage = path_info['percentage']
            
            # Create flow text
            flow_text = ' → '.join(sequence[:5])  # Limit to first 5 steps
            if len(sequence) > 5:
                flow_text += f' → ... ({len(sequence)} steps total)'
            
            label = f"Path {i+1}: {frequency} traces ({percentage:.1f}%)"
            
            # Plot as horizontal bar
            y_pos = top_n - i - 1
            ax.barh(y_pos, percentage, color=sns.color_palette("husl", top_n)[i])
            ax.text(percentage + 1, y_pos, label, va='center', fontweight='bold')
            ax.text(1, y_pos - 0.3, flow_text, va='top', fontsize=9, style='italic')
        
        ax.set_xlabel('Percentage of Traces (%)', fontweight='bold')
        ax.set_title(f'Top {top_n} Most Frequent Process Paths (Happy Paths)', 
                    fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlim(0, max(p['percentage'] for p in happy_paths[:top_n]) * 1.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'process_flow.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'process_flow.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_bottlenecks(self, top_n=10, save=True, show=True):
        """PM7: Bottleneck analysis."""
        self._check_matplotlib()
        
        bottlenecks = self.ip.pm7_bottlenecks()
        
        if not bottlenecks or 'top_10_bottlenecks' not in bottlenecks:
            print("⚠️ No bottleneck data available")
            return
        
        top_bottlenecks = bottlenecks['top_10_bottlenecks'][:top_n]
        
        activities = [b['activity'] for b in top_bottlenecks]
        mean_durations = [b['mean_duration'] for b in top_bottlenecks]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = sns.color_palette("Reds_r", len(activities))
        bars = ax.barh(range(len(activities)), mean_durations, color=colors)
        
        ax.set_yticks(range(len(activities)))
        ax.set_yticklabels(activities)
        ax.set_xlabel('Mean Duration (seconds)', fontweight='bold')
        ax.set_title(f'Top {len(activities)} Process Bottlenecks', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add duration labels
        for i, (bar, duration) in enumerate(zip(bars, mean_durations)):
            ax.text(duration, i, f' {duration:.2f}s', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'bottlenecks.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'bottlenecks.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_success_vs_failure(self, save=True, show=True):
        """PM8: Success vs failure path comparison."""
        self._check_matplotlib()
        
        comparison = self.ip.pm8_success_vs_failure_paths()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Success vs Failure Process Comparison', fontsize=16, fontweight='bold')
        
        # Success traces
        success_count = comparison['successful_traces']['count']
        success_variants = comparison['successful_traces']['unique_variants']
        
        axes[0].bar(['Total Traces', 'Unique Variants'], 
                   [success_count, success_variants],
                   color=['green', 'lightgreen'])
        axes[0].set_title('Successful Traces', fontweight='bold', color='green')
        axes[0].set_ylabel('Count')
        
        for i, v in enumerate([success_count, success_variants]):
            axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Failure traces
        failure_count = comparison['failed_traces']['count']
        failure_variants = comparison['failed_traces']['unique_variants']
        
        axes[1].bar(['Total Traces', 'Unique Variants'], 
                   [failure_count, failure_variants],
                   color=['red', 'lightcoral'])
        axes[1].set_title('Failed Traces', fontweight='bold', color='red')
        axes[1].set_ylabel('Count')
        
        for i, v in enumerate([failure_count, failure_variants]):
            axes[1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'success_vs_failure.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'success_vs_failure.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_behavioral_clusters(self, n_clusters=3, save=True, show=True):
        """PM10: Behavioral cluster visualization."""
        self._check_matplotlib()
        
        try:
            clusters = self.ip.pm10_behavioral_clusters(n_clusters=n_clusters)
        except Exception as e:
            print(f"⚠️ Error generating clusters: {e}")
            return
        
        if 'error' in clusters:
            print(f"⚠️ {clusters['error']}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cluster_ids = []
        trace_counts = []
        tool_usage_rates = []
        
        for cid, info in clusters['clusters'].items():
            cluster_ids.append(f"Cluster {cid}")
            trace_counts.append(info['trace_count'])
            tool_usage_rates.append(info['tool_usage_rate'] * 100)
        
        x = np.arange(len(cluster_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, trace_counts, width, label='Trace Count', color='skyblue')
        bars2 = ax.bar(x + width/2, tool_usage_rates, width, label='Tool Usage %', color='orange')
        
        ax.set_xlabel('Clusters', fontweight='bold')
        ax.set_title(f'Behavioral Clusters (n={n_clusters})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_ids)
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'behavioral_clusters.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'behavioral_clusters.png'}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # ========== INTERACTIVE VISUALIZATIONS (Plotly) ==========
    
    def create_interactive_dashboard(self, save=True):
        """Create an interactive HTML dashboard with all key metrics."""
        self._check_plotly()
        
        # Get data
        summary = self.ip.ds1_semantic_summary()
        tool_freq = self.ip.ds6_tool_frequency()
        token_dist = self.ip.ds7_token_distribution()
        success_failure = self.ip.ds9_success_failure_rate()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Token Distribution', 'Tool Usage', 
                          'Trace Outcomes', 'Session Statistics'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'indicator'}]]
        )
        
        # 1. Token distribution pie chart
        fig.add_trace(
            go.Pie(
                labels=['Prompt', 'Completion', 'Thoughts'],
                values=[
                    token_dist['prompt_tokens']['total'],
                    token_dist['completion_tokens']['total'],
                    token_dist['thoughts_tokens']['total']
                ],
                name='Tokens'
            ),
            row=1, col=1
        )
        
        # 2. Tool usage bar chart
        if tool_freq:
            sorted_tools = sorted(tool_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            tools, counts = zip(*sorted_tools)
            
            fig.add_trace(
                go.Bar(x=list(tools), y=list(counts), name='Tool Calls'),
                row=1, col=2
            )
        
        # 3. Success vs Failure
        fig.add_trace(
            go.Bar(
                x=['Successful', 'Failed'],
                y=[success_failure['successful_traces'], success_failure['failed_traces']],
                marker_color=['green', 'red'],
                name='Outcomes'
            ),
            row=2, col=1
        )
        
        # 4. Key metric indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=summary['total_events'],
                title="Total Events",
                delta={'reference': 0},
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Event Log Interactive Dashboard",
            showlegend=False,
            height=800
        )
        
        if save:
            output_file = self.output_dir / 'interactive_dashboard.html'
            fig.write_html(str(output_file))
            print(f"✅ Saved: {output_file}")
        
        return fig
    
    # ========== COMPREHENSIVE REPORT ==========
    
    def generate_all_visualizations(self, show=False):
        """Generate all available visualizations."""
        print("🎨 Generating comprehensive visualization report...")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        
        try:
            print("1️⃣  Session Overview...")
            self.plot_session_overview(save=True, show=show)
            
            print("2️⃣  Tool Usage...")
            self.plot_tool_usage(save=True, show=show)
            
            print("3️⃣  Token Consumption...")
            self.plot_token_consumption(save=True, show=show)
            
            print("4️⃣  Duration Distributions...")
            self.plot_duration_distributions(save=True, show=show)
            
            print("5️⃣  Correlation Heatmap...")
            self.plot_correlation_heatmap(save=True, show=show)
            
            print("6️⃣  Process Flow...")
            self.plot_process_flow(save=True, show=show)
            
            print("7️⃣  Bottlenecks...")
            self.plot_bottlenecks(save=True, show=show)
            
            print("8️⃣  Success vs Failure...")
            self.plot_success_vs_failure(save=True, show=show)
            
            print("9️⃣  Behavioral Clusters...")
            self.plot_behavioral_clusters(save=True, show=show)
            
            if PLOTLY_AVAILABLE:
                print("🔟 Interactive Dashboard...")
                self.create_interactive_dashboard(save=True)
            
            print()
            print(f"✅ All visualizations saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()


# Usage example
if __name__ == "__main__":
    from event_processor import EventProcessor, InsightProcessor
    
    # Initialize
    ep = EventProcessor(log_dir=".log")
    ip = InsightProcessor(ep)
    ip.load_and_structure()
    
    # Create visualizer
    viz = EventVisualizer(ip)
    
    # Generate all visualizations
    viz.generate_all_visualizations(show=False)
    
    print("🎉 Visualization complete!")