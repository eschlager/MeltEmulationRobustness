## extra plotting routines

import re
import numpy as np
import matplotlib as mpl

def contains_dataset_token(d, m):
    return re.search(rf'(?<![A-Za-z0-9]){re.escape(str(d))}(?![A-Za-z0-9])', str(m)) is not None

def _assign_patches_to_xticks(ax, bars):
    """
    Assign each bar patch to an xtick index by binning patch centers between
    midpoints of xticks. Returns dict: xtick_index -> list of (patch, center)
    """
    xticks = np.array(ax.get_xticks())
    if xticks.size == 0:
        return {}

    # compute midpoints between xticks to form bin boundaries
    if xticks.size == 1:
        # everything belongs to bin 0
        boundaries = np.array([])
    else:
        boundaries = (xticks[:-1] + xticks[1:]) / 2.0

    centers = np.array([p.get_x() + p.get_width() / 2.0 for p in bars])

    # digitize centers into bins 0..len(xticks)-1
    if boundaries.size:
        bin_indices = np.digitize(centers, boundaries)
    else:
        bin_indices = np.zeros_like(centers, dtype=int)

    groups = {}
    for idx, bi in enumerate(bin_indices):
        groups.setdefault(int(bi), []).append((bars[idx], centers[idx]))

    # sort each group's patches by center (left -> right)
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda pc: pc[1])

    return groups

def hatch_bar(ax, plot_df, debug=False):
    x_order = list(plot_df["dataset"].cat.categories)
    hue_order = list(plot_df["model"].cat.categories)

    bars = [p for p in ax.patches if isinstance(p, mpl.patches.Rectangle)]
    if not bars:
        if debug:
            print("hatch_bar: no rectangle patches found")
        return ax

    groups = _assign_patches_to_xticks(ax, bars)

    for xi, d in enumerate(x_order):
        # skip if no group (e.g., more categories than xticks)
        if xi not in groups:
            if debug:
                print(f"hatch_bar: no patches for xtick {xi} (dataset {d})")
            continue

        ps = groups[xi]  # list of (patch, center) sorted by center
        present_models = [
            m for m in hue_order
            if ((plot_df["dataset"] == d) & (plot_df["model"] == m)).any()
        ]

        if len(ps) != len(present_models) and debug:
            print(f"hatch_bar: mismatch at dataset {d}: patches={len(ps)} present_models={len(present_models)}")

        # map left-to-right patches to present_models in seaborn hue order
        for (p, _center), m in zip(ps, present_models):
            ms = str(m).strip()
            ds = str(d).strip()
            if ms == ds:
                p.set_hatch("///")
                p.set_edgecolor("black")
                p.set_linewidth(1.0)
                p.set_zorder(5)
            elif contains_dataset_token(ds, ms):
                p.set_hatch("...")
                p.set_edgecolor("black")
                p.set_linewidth(1.0)
                p.set_zorder(5)

    # draw error bars on top
    for line in ax.lines:
        line.set_zorder(10)
        line.set_color("black")
        line.set_linewidth(1.2)
    return ax



def best_iter_scatter(ax, plot_df, metric, debug=False):
    """
    Robust placement of best-iter 'x' markers:
      - groups patches into xtick bins
      - for each dataset/model use the plotted bar height (from plot_df) to choose the matching patch
      - fallback to left->right mapping when needed
    """
    import numpy as np
    import matplotlib as mpl

    # helper: bin patches by xtick midpoints (same as other helpers)
    def _assign_patches_to_xticks(ax, bars):
        xticks = np.array(ax.get_xticks())
        if xticks.size == 0:
            return {}
        if xticks.size == 1:
            boundaries = np.array([])
        else:
            boundaries = (xticks[:-1] + xticks[1:]) / 2.0
        centers = np.array([p.get_x() + p.get_width() / 2.0 for p in bars])
        if boundaries.size:
            bin_indices = np.digitize(centers, boundaries)
        else:
            bin_indices = np.zeros_like(centers, dtype=int)
        groups = {}
        for idx, bi in enumerate(bin_indices):
            groups.setdefault(int(bi), []).append((bars[idx], centers[idx]))
        # sort patches in each group by center (left -> right)
        for k in list(groups.keys()):
            groups[k] = sorted(groups[k], key=lambda pc: pc[1])
        return groups

    x_order = list(plot_df["dataset"].cat.categories)
    hue_order = list(plot_df["model"].cat.categories)

    # only rectangle patches
    bars = [p for p in ax.patches if isinstance(p, mpl.patches.Rectangle)]
    if not bars:
        if debug:
            print("No bar patches found")
        return ax

    groups = _assign_patches_to_xticks(ax, bars)
    best_df = plot_df.loc[plot_df["best_iter"].astype(bool)]

    # precompute patch centers and tops
    patch_centers = np.array([p.get_x() + p.get_width() / 2.0 for p in bars])
    patch_tops = np.array([p.get_y() + p.get_height() for p in bars])
    patch_widths = np.array([p.get_width() for p in bars])
    median_width = float(np.median(patch_widths)) if patch_widths.size else 0.0
    eps = 1e-12

    for xi, d in enumerate(x_order):
        # skip if no xtick (defensive)
        xticks = np.array(ax.get_xticks())
        if xi >= len(xticks):
            if debug:
                print(f"Skipping dataset {d}: no xtick index {xi}")
            continue
        xt = xticks[xi]

        present_models = [
            m for m in hue_order
            if ((plot_df["dataset"] == d) & (plot_df["model"] == m)).any()
        ]
        if not present_models:
            if debug:
                print(f"No present models for dataset {d}")
            continue

        # get patches in this cluster
        if xi not in groups or len(groups[xi]) == 0:
            if debug:
                print(f"No patch group found for xtick {xi} (dataset {d})")
            continue

        cluster = groups[xi]               # list of (patch, center) sorted by center
        cluster_patches = [it[0] for it in cluster]
        cluster_centers = np.array([it[1] for it in cluster])
        cluster_indices_global = [bars.index(p) for p in cluster_patches]

        # For each present model, compute the plotted bar median (bar top in data coords)
        # The barplot used estimator="median" on the same plot_df values, so we can derive it directly.
        for local_idx, m in enumerate(present_models):
            # median bar value (height) as plotted for this dataset/model
            rows = plot_df[(plot_df["dataset"] == d) & (plot_df["model"] == m)]
            if rows.shape[0] == 0:
                if debug:
                    print(f"No plot_df rows for {d}/{m}")
                continue

            # If multiple rows (multiple iter?), barplot's estimator over the underlying distribution
            # used to produce the bar: but in your generation you already computed the values used
            # in the dataframe rows (they were per run). We want the median of those values (same as seaborn estimator)
            expected_bar_height = float(rows[metric].median())

            # find the patch in this cluster whose top is closest to expected_bar_height
            tops_in_cluster = patch_tops[cluster_indices_global]
            top_diffs = np.abs(tops_in_cluster - expected_bar_height)

            # If a clear best by top difference exists (significantly smaller than others), use it.
            best_rel_idx = int(np.argmin(top_diffs))
            best_global_idx = cluster_indices_global[best_rel_idx]
            best_patch = bars[best_global_idx]
            best_center = patch_centers[best_global_idx]
            best_top_diff = top_diffs[best_rel_idx]

            if debug:
                print(f"Dataset={d} Model={m}: expected_y={expected_bar_height:.6g}")
                for j, gi in enumerate(cluster_indices_global):
                    print(f"  patch {j}: center={patch_centers[gi]:.4f}, top={patch_tops[gi]:.6g}, top_diff={top_diffs[j]:.6g}")

            # decide acceptance threshold: relative to expected_bar_height or small absolute if near-zero
            rel_tol = 0.25  # 25% relative tolerance
            abs_tol = 1e-8  # absolute tolerance fallback
            tol = max(rel_tol * (abs(expected_bar_height) + eps), abs_tol)

            if best_top_diff <= tol:
                # good match by height — place marker at that patch center
                chosen_center = best_center
                if debug:
                    print(f"  -> chosen by height: patch idx {best_global_idx} center {chosen_center:.4f} top_diff {best_top_diff:.6g}")
            else:
                # fallback: try mapping by left->right order between cluster patches and present_models
                # map by index within cluster (if counts match) else choose nearest center to estimated order
                if len(cluster_patches) == len(present_models):
                    # find index of this model within present_models and pick that patch
                    idx_in_present = present_models.index(m)
                    chosen_patch = cluster_patches[idx_in_present]
                    chosen_center = chosen_patch.get_x() + chosen_patch.get_width() / 2.0
                    if debug:
                        print(f"  -> fallback left->right mapping idx {idx_in_present}, center {chosen_center:.4f}")
                else:
                    # choose patch whose center is nearest to xt + local offset (use local ordering)
                    # compute local offsets assuming equal spacing centered on xt
                    n_local = len(present_models)
                    offsets = (np.arange(n_local) - (n_local - 1) / 2.0) * (median_width if median_width > 0 else 1.0)
                    expected_x = xt + offsets[local_idx]
                    center_diffs = np.abs(cluster_centers - expected_x)
                    ci = int(np.argmin(center_diffs))
                    chosen_center = cluster_centers[ci]
                    if debug:
                        print(f"  -> fallback nearest-center expected_x {expected_x:.4f}, chosen center {chosen_center:.4f}")

            # Now place the scatter for the best-iter row corresponding to this (dataset,model)
            row = best_df[
                (best_df["dataset"] == d) &
                (best_df["model"] == m)
            ]
            if row.shape[0] == 0:
                if debug:
                    print(f"  no best-row for {d}/{m}, skipping")
                continue

            y = float(row[metric].iloc[0])
            ax.scatter(chosen_center, y, color='black', marker='x', s=40, zorder=20)
            if debug:
                print(f"  placed x at x={chosen_center:.4f}, y={y:.6g}")

    return ax


def _assign_patches_to_xticks(ax, bars):
    xticks = np.array(ax.get_xticks())
    if xticks.size == 0:
        return {}
    if xticks.size == 1:
        boundaries = np.array([])
    else:
        boundaries = (xticks[:-1] + xticks[1:]) / 2.0
    centers = np.array([p.get_x() + p.get_width() / 2.0 for p in bars])
    if boundaries.size:
        bin_indices = np.digitize(centers, boundaries)
    else:
        bin_indices = np.zeros_like(centers, dtype=int)
    groups = {}
    for idx, bi in enumerate(bin_indices):
        groups.setdefault(int(bi), []).append((bars[idx], centers[idx]))
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda pc: pc[1])
    return groups

def draw_brackets_instead_of_ticks(
    ax,
    tick_height=0.03,        # height of vertical tick in axes-fraction units
    linewidth=1.5,
    color='black',
    bracket_mode='group',    # 'group' or 'fixed'
    fixed_frac=0.25,        # fraction of xtick spacing when bracket_mode == 'fixed'
    pad_bottom=0.06,        # extra bottom margin (axes fraction)
    label_pad_pts=8,        # move xtick labels down (points)
    debug=False
):
    """
    Draw upward-facing brackets directly below the x-axis, then move xtick labels below the brackets.
    - Vertical ticks go from y = -tick_height up to y = 0 (axis line) so they meet the axis.
    - Horizontal connector is at y = -tick_height.
    - Call after plotting & any hatch/markers.
    """
    # hide default tick marks but keep labels; move labels down
    ax.tick_params(axis='x', which='both', length=0, pad=label_pad_pts)

    bars = [p for p in ax.patches if isinstance(p, mpl.patches.Rectangle)]
    if not bars:
        if debug:
            print("draw_brackets_below_axis_facing_up: no bar patches found")
        return ax

    groups = _assign_patches_to_xticks(ax, bars)
    xticks = np.array(ax.get_xticks())

    # fallback xtick spacing for 'fixed' mode
    if xticks.size > 1:
        xt_spacing = np.min(np.diff(xticks))
    else:
        all_widths = np.array([p.get_width() for p in bars])
        xt_spacing = np.median(all_widths) * 2 if all_widths.size else 1.0

    transform = ax.get_xaxis_transform()  # x in data coords, y in axes-fraction coords
    fig = ax.get_figure()

    # ensure there's enough bottom margin to show bracket + labels
    # need = top of bracket distance below axis (tick_height) + some pad for labels
    need = tick_height + pad_bottom
    current_bottom = fig.subplotpars.bottom
    if need > current_bottom:
        fig.subplots_adjust(bottom=need)

    y_top = 0.0                    # axis line in axes fraction
    y_bottom_of_ticks = -tick_height
    y_connector = y_bottom_of_ticks  # horizontal connector sits at bottom of ticks

    for xi in range(len(xticks)):
        # compute bracket halfwidth and center
        if xi in groups and len(groups[xi]) > 0:
            patches = [it[0] for it in groups[xi]]
            left = min(p.get_x() for p in patches)
            right = max(p.get_x() + p.get_width() for p in patches)
            center = (left + right) / 2.0
            group_half = (right - left) / 2.0
        else:
            center = xticks[xi]
            group_half = 0.0

        if bracket_mode == 'group' and group_half > 0:
            halfwidth = group_half
        else:
            halfwidth = 0.5 * fixed_frac * xt_spacing

        left_x = center - halfwidth
        right_x = center + halfwidth

        # draw vertical ticks upward: from y_bottom_of_ticks up to y_top (0.0)
        ax.plot([left_x, left_x], [y_bottom_of_ticks, y_top],
                transform=transform, color=color, linewidth=linewidth,
                solid_capstyle='butt', zorder=30, clip_on=False)
        ax.plot([right_x, right_x], [y_bottom_of_ticks, y_top],
                transform=transform, color=color, linewidth=linewidth,
                solid_capstyle='butt', zorder=30, clip_on=False)
        # draw horizontal connector at bottom of ticks (bracket line)
        ax.plot([left_x, right_x], [y_connector, y_connector],
                transform=transform, color=color, linewidth=linewidth,
                solid_capstyle='butt', zorder=30, clip_on=False)

        if debug:
            print(f"xtick idx={xi}: center={center:.3f}, left={left_x:.3f}, right={right_x:.3f}, tick_height={tick_height:.3f}")

    return ax