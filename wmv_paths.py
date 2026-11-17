"""WM-Verdict path module (release 2026-11-16). Every machine path in the code goes through here.

Environment variables (a placeholder that is not set stays literally in the path, so a missing setting shows up as a
"no such file: $VAR/..." error; set WMV_STRICT_PATHS=1 to raise immediately instead):
  $WMV_ROOT                 data root in the legacy layout (libero/out/..., protocol.yaml, banks); build it with scripts/make_data_root.py
  $WMV_WEB_ROOT             local clone of the annotation-website repository
  $WMV_NAS                  shared storage root of the original runs (cloud pulls, node_sync); only for provenance scripts
  $NAS                      shared storage mount point (provenance scripts only)
  $LEROBOT_ROOT             lerobot clone at e624f3f7 (only if it is not pip-installed)
  $MMBENCH2_ROOT            MMBench2 pilot directory (not shipped)
  $DINO_WM_ROOT             DINO-WM clone (not shipped)
  $COSMOS_DATA_ROOT         collaborators' Cosmos Policy / VPP2 audit data root
  $COSMOS_DATA_HOST         collaborator storage root (Cosmos / VPP2)
  $UVA_DATA_ROOT            collaborators' UVA validation data root
  $UVA_DATA_HOST            collaborator storage root (UVA)
  $HOME                     home directory
  $CONDA_ROOT               conda installation root (envs lerobot_libero, openwam)
  $WMV_VENV_ROOT            directory holding venv_pinned (B-family corruption worker)
  $SCRATCH                  scratch directory
  $LIBERO_REPO              a LIBERO source checkout (BDDL files)
  $OPENWAM_REPO             OpenWAM clone at 1e103e1
  $WORLDVLA_REPO            WorldVLA clone (branch worldvla @ ebdc36dc5cfd)
  $LIBERO_PLUS_ENV_WRAPPER  LIBERO-plus env_wrapper.py @ 4976dc30 (third_party/fetch_libero_plus.sh)
  $DROPBOX_REMOTE           rclone remote of the data mirror (provenance scripts only)
  $NODE_CLOUD               directory name (placeholder) of the cloud node's pulled results

  P("$WMV_ROOT/libero/out/runs")   -> expanded path
  code("libero/openwam/ow_runner.py") -> location of that (legacy-layout) code file inside this repository
  bootstrap()                      -> put every code directory of this repository on sys.path (idempotent)
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PATH_VARS = ('WMV_ROOT', 'WMV_WEB_ROOT', 'WMV_NAS', 'NAS', 'LEROBOT_ROOT', 'MMBENCH2_ROOT', 'DINO_WM_ROOT', 'COSMOS_DATA_ROOT', 'COSMOS_DATA_HOST', 'UVA_DATA_ROOT', 'UVA_DATA_HOST', 'HOME', 'CONDA_ROOT', 'WMV_VENV_ROOT', 'SCRATCH', 'LIBERO_REPO', 'OPENWAM_REPO', 'WORLDVLA_REPO', 'LIBERO_PLUS_ENV_WRAPPER', 'DROPBOX_REMOTE', 'NODE_CLOUD')
_RX = re.compile(r"\$(\{)?(" + "|".join(sorted(PATH_VARS, key=len, reverse=True)) + r")\b(?(1)\})")
LEGACY_CODE = {
    'libero/agent_human_agreement.py': 'analysis/cells/agent_human_agreement.py',
    'libero/anchor_bank.py': 'protocol/core/anchor_bank.py',
    'libero/anchor_bank_t2.py': 'protocol/core/anchor_bank_t2.py',
    'libero/anchor_bank_tx.py': 'protocol/core/anchor_bank_tx.py',
    'libero/attrib_sheets.py': 'analysis/cells/attrib_sheets.py',
    'libero/attrib_zoom.py': 'analysis/cells/attrib_zoom.py',
    'libero/backfill/famb_t2_capture.py': 'protocol/runners/lingbot/famb_t2_capture.py',
    'libero/backfill/lb_capture.py': 'protocol/runners/lingbot/lb_capture.py',
    'libero/backfill/ow_backfill.py': 'protocol/runners/openwam/ow_backfill.py',
    'libero/bfam/bcorrupt.py': 'perturbations/bfam/bcorrupt.py',
    'libero/bfam/bcorrupt_server.py': 'perturbations/bfam/bcorrupt_server.py',
    'libero/bfam/bgate.py': 'protocol/checks/bgate.py',
    'libero/bfam/bgate_traj.py': 'protocol/checks/bgate_traj.py',
    'libero/bfam/bpreflight.py': 'protocol/checks/bpreflight.py',
    'libero/bfam/lingbot_famB.py': 'protocol/runners/lingbot/lingbot_famB.py',
    'libero/bfam/lingbot_tx_famB.py': 'protocol/runners/lingbot/lingbot_tx_famB.py',
    'libero/bfam/ow_famB.py': 'protocol/runners/openwam/ow_famB.py',
    'libero/bfam/ow_tx_famB.py': 'protocol/runners/openwam/ow_tx_famB.py',
    'libero/cells_final.py': 'analysis/cells/cells_final.py',
    'libero/compare_models.py': 'analysis/cells/compare_models.py',
    'libero/compare_new.py': 'analysis/cells/compare_new.py',
    'libero/content_review.py': 'analysis/cells/content_review.py',
    'libero/familyA_bank.py': 'perturbations/init_state/familyA_bank.py',
    'libero/familyA_ik.py': 'perturbations/init_state/familyA_ik.py',
    'libero/familyA_protocol.py': 'perturbations/init_state/familyA_protocol.py',
    'libero/familyA_t2_bank.py': 'perturbations/init_state/familyA_t2_bank.py',
    'libero/familyA_tx_bank.py': 'perturbations/init_state/familyA_tx_bank.py',
    'libero/familyP_bank.py': 'perturbations/init_state/familyP_bank.py',
    'libero/familyP_protocol.py': 'perturbations/init_state/familyP_protocol.py',
    'libero/familyP_t2_bank.py': 'perturbations/init_state/familyP_t2_bank.py',
    'libero/familyP_tx_bank.py': 'perturbations/init_state/familyP_tx_bank.py',
    'libero/ffam/f_cases.py': 'analysis/cells/f_cases.py',
    'libero/ffam/f_cases_cmp.py': 'analysis/cells/f_cases_cmp.py',
    'libero/ffam/f_pred.py': 'analysis/cells/f_pred.py',
    'libero/ffam/f_state_probe.py': 'analysis/cells/f_state_probe.py',
    'libero/ffam/f_succ_ref.py': 'analysis/cells/f_succ_ref.py',
    'libero/ffam/ffam_def.py': 'perturbations/ffam/ffam_def.py',
    'libero/ffam/hc_lingbot.py': 'protocol/runners/lingbot/hc_lingbot.py',
    'libero/ffam/hc_openwam.py': 'protocol/runners/openwam/hc_openwam.py',
    'libero/ffam/lingbot_famF.py': 'protocol/runners/lingbot/lingbot_famF.py',
    'libero/ffam/ow_famF.py': 'protocol/runners/openwam/ow_famF.py',
    'libero/gen_timing_rule.py': 'predicates/lingbot/gen_timing_rule.py',
    'libero/lingbot_baseline.py': 'protocol/runners/lingbot/lingbot_baseline.py',
    'libero/lingbot_famA.py': 'protocol/runners/lingbot/lingbot_famA.py',
    'libero/lingbot_predicate.py': 'predicates/lingbot/lingbot_predicate.py',
    'libero/lingbot_t2_closed.py': 'protocol/runners/lingbot/lingbot_t2_closed.py',
    'libero/lingbot_t2_cloud.py': 'protocol/runners/lingbot/lingbot_t2_cloud.py',
    'libero/lingbot_tx_closed.py': 'protocol/runners/lingbot/lingbot_tx_closed.py',
    'libero/lingbot_tx_famO_closed.py': 'protocol/runners/lingbot/lingbot_tx_famO_closed.py',
    'libero/lingbot_tx_run.py': 'protocol/runners/lingbot/lingbot_tx_run.py',
    'libero/lingbot_unit.py': 'protocol/runners/lingbot/lingbot_unit.py',
    'libero/machine_probe.py': 'protocol/core/machine_probe.py',
    'libero/min_artifacts.py': 'protocol/core/min_artifacts.py',
    'libero/native_env.py': 'perturbations/init_state/native_env.py',
    'libero/openwam/fit_controller.py': 'protocol/runners/openwam/fit_controller.py',
    'libero/openwam/libero_lang.py': 'protocol/runners/openwam/libero_lang.py',
    'libero/openwam/openwam_unit.py': 'protocol/runners/openwam/openwam_unit.py',
    'libero/openwam/ow_cl_rerun.py': 'protocol/runners/openwam/ow_cl_rerun.py',
    'libero/openwam/ow_freeze.py': 'predicates/openwam/ow_freeze.py',
    'libero/openwam/ow_health.py': 'protocol/runners/openwam/ow_health.py',
    'libero/openwam/ow_predicate.py': 'predicates/openwam/ow_predicate.py',
    'libero/openwam/ow_runner.py': 'protocol/runners/openwam/ow_runner.py',
    'libero/openwam/ow_score.py': 'predicates/openwam/ow_score.py',
    'libero/openwam/ow_t3_review.py': 'predicates/openwam/ow_t3_review.py',
    'libero/openwam/ow_t5_contentcheck.py': 'predicates/openwam/ow_t5_contentcheck.py',
    'libero/openwam/ow_t5_hn.py': 'predicates/openwam/ow_t5_hn.py',
    'libero/openwam/ow_t5_v2.py': 'predicates/openwam/ow_t5_v2.py',
    'libero/openwam/ow_view.py': 'protocol/runners/openwam/ow_view.py',
    'libero/openwam/setup_openwam.sh': 'scripts/setup_openwam.sh',
    'libero/openwam/sim_server.py': 'protocol/runners/openwam/sim_server.py',
    'libero/out/annotation/render_cosmos_r2.py': 'annotation_pipeline/round2/render_cosmos_r2.py',
    'libero/out/annotation/round2_real/build_manifest.py': 'annotation_pipeline/round2/build_manifest.py',
    'libero/out/annotation/round2_real/export_webp.py': 'annotation_pipeline/round2/export_webp.py',
    'libero/out/annotation/round2_real/ours_cards.py': 'annotation_pipeline/round2/ours_cards.py',
    'libero/out/annotation/round2_real/real_cards.py': 'annotation_pipeline/round2/real_cards.py',
    'libero/out/annotation/round3_imagined/r3_cards.py': 'annotation_pipeline/round3/r3_cards.py',
    'libero/out/annotation/samecase_20261115/build_cards.py': 'annotation_pipeline/build_cards.py',
    'libero/out/annotation/samecase_20261115/check_pages.py': 'annotation_pipeline/check_pages.py',
    'libero/out/annotation/samecase_20261115/check_real_consistency.py': 'annotation_pipeline/check_real_consistency.py',
    'libero/out/annotation/samecase_20261115/inventory.py': 'annotation_pipeline/inventory.py',
    'libero/out/annotation/samecase_20261115/samecase_replay.py': 'annotation_pipeline/samecase_replay.py',
    'libero/out/annotation/samecase_20261115/select_pages.py': 'annotation_pipeline/select_pages.py',
    'libero/out/artifact_verify/neg_test.py': 'protocol/checks/neg_test.py',
    'libero/out/b_family/healthcheck/bcheck.py': 'perturbations/bfam/healthcheck/bcheck.py',
    'libero/out/b_family/healthcheck/bcheck_extra.py': 'perturbations/bfam/healthcheck/bcheck_extra.py',
    'libero/out/b_family/healthcheck/dose_plot.py': 'perturbations/bfam/healthcheck/dose_plot.py',
    'libero/out/b_family/healthcheck/mkvenv.sh': 'perturbations/bfam/healthcheck/mkvenv.sh',
    'libero/out/content_review/lingbot_famO/make_cards.py': 'analysis/cells/make_cards.py',
    'libero/out/content_review/lingbot_famO/make_subset.py': 'analysis/cells/make_subset.py',
    'libero/out/content_review/openwam_t3/t3_controls.py': 'analysis/cells/t3_controls.py',
    'libero/out/optimism/cases.py': 'analysis/optimism/cases.py',
    'libero/out/optimism/judge_ow_t1.py': 'analysis/optimism/judge_ow_t1.py',
    'libero/out/optimism/opt_coverage.py': 'analysis/optimism/opt_coverage.py',
    'libero/out/optimism/prescreen/cosmos2_prep.py': 'analysis/optimism/cosmos2_prep.py',
    'libero/out/optimism/prescreen/cosmos_full_prep.py': 'analysis/optimism/cosmos_full_prep.py',
    'libero/out/optimism/prescreen/cosmos_prep.py': 'analysis/optimism/cosmos_prep.py',
    'libero/out/optimism/prescreen/decode_pkg/decode_last_frame.py': 'analysis/optimism/decode_last_frame.py',
    'libero/out/optimism/prescreen/decode_pkg/decode_last_frame_cpu.py': 'analysis/optimism/decode_last_frame_cpu.py',
    'libero/out/optimism/prescreen/lb2_prep.py': 'analysis/optimism/lb2_prep.py',
    'libero/out/optimism/prescreen/prep.py': 'analysis/optimism/prep.py',
    'libero/out/optimism/prescreen/prescreen_report.py': 'analysis/optimism/prescreen_report.py',
    'libero/out/optimism/prescreen/record.py': 'analysis/optimism/record.py',
    'libero/out/optimism/prescreen/unified_report.py': 'analysis/optimism/unified_report.py',
    'libero/out/optimism/rank_more.py': 'analysis/optimism/rank_more.py',
    'libero/out/optimism/rank_view.py': 'analysis/optimism/rank_view.py',
    'libero/out/optimism/report.py': 'analysis/optimism/report.py',
    'libero/out/optimism/report_full.py': 'analysis/optimism/report_full.py',
    'libero/out/optimism/sharpness/sharp_report.py': 'analysis/optimism/sharp_report.py',
    'libero/out/optimism/sharpness/sharpness.py': 'analysis/optimism/sharpness.py',
    'libero/out/optimism/sharpness/verify2.py': 'analysis/optimism/verify2.py',
    'libero/out/results_page/data_audit_20261115/apply_fixes.py': 'analysis/results_page/audit/apply_fixes.py',
    'libero/out/results_page/data_audit_20261115/audit_lib.py': 'analysis/results_page/audit/audit_lib.py',
    'libero/out/results_page/data_audit_20261115/check_media.py': 'analysis/results_page/audit/check_media.py',
    'libero/out/results_page/data_audit_20261115/compare_page.py': 'analysis/results_page/audit/compare_page.py',
    'libero/out/results_page/data_audit_20261115/crosscheck_teammate.py': 'analysis/results_page/audit/crosscheck_teammate.py',
    'libero/out/results_page/data_audit_20261115/diff_check.py': 'analysis/results_page/audit/diff_check.py',
    'libero/out/results_page/data_audit_20261115/make_reports.py': 'analysis/results_page/audit/make_reports.py',
    'libero/out/results_page/data_audit_20261115/readback.py': 'analysis/results_page/audit/readback.py',
    'libero/out/results_page/data_audit_20261115/texts.py': 'analysis/results_page/audit/texts.py',
    'libero/out/results_page/data_audit_20261115/verify_sources.py': 'analysis/results_page/audit/verify_sources.py',
    'libero/out/results_page/make_page.py': 'analysis/results_page/make_page.py',
    'libero/out/rule_v2/diag_false_alarms.py': 'predicates/rules/diag_false_alarms.py',
    'libero/out/task2/build_uva_index.py': 'analysis/main_table/build_uva_index.py',
    'libero/out/task2/cat_summary.py': 'analysis/main_table/cat_summary.py',
    'libero/out/task2/diag_ow_t2.py': 'analysis/diagnostics/diag_ow_t2.py',
    'libero/out/task2/ext_models.py': 'analysis/main_table/ext_models.py',
    'libero/out/task2/fk_compare.py': 'analysis/main_table/fk_compare.py',
    'libero/out/task2/gen_closed_main.py': 'analysis/main_table/gen_closed_main.py',
    'libero/out/task2/insight_recheck.py': 'analysis/main_table/insight_recheck.py',
    'libero/out/task2/make_snapshot.py': 'analysis/main_table/make_snapshot.py',
    'libero/out/task2/make_target_list.py': 'analysis/main_table/make_target_list.py',
    'libero/out/task2/no_nlw_views.py': 'analysis/main_table/no_nlw_views.py',
    'libero/out/task2/ow_tx_extra.py': 'protocol/runners/openwam/ow_tx_extra.py',
    'libero/out/task2/ow_tx_run.py': 'protocol/runners/openwam/ow_tx_run.py',
    'libero/out/task2/probe_knob.py': 'analysis/diagnostics/probe_knob.py',
    'libero/out/task2/smoke_lingbot.py': 'protocol/runners/lingbot/smoke_lingbot.py',
    'libero/out/task2/smoke_openwam.py': 'protocol/runners/openwam/smoke_openwam.py',
    'libero/out/task2/t2_bridge/bridge_t2.py': 'protocol/checks/bridge_t2.py',
    'libero/out/task2/uva_k_scan.py': 'analysis/main_table/uva_k_scan.py',
    'libero/out/task2/uva_t01_scan.py': 'analysis/main_table/uva_t01_scan.py',
    'libero/out/task_prereview/prereview.py': 'protocol/checks/prereview.py',
    'libero/out/verify_20261114/verify_independent.py': 'protocol/checks/verify_independent.py',
    'libero/phys/lingbot_famO.py': 'protocol/runners/lingbot/lingbot_famO.py',
    'libero/phys/ow_egn_truth.py': 'protocol/runners/openwam/ow_egn_truth.py',
    'libero/phys/ow_famO.py': 'protocol/runners/openwam/ow_famO.py',
    'libero/phys/ow_phys_closed_tx.py': 'protocol/runners/openwam/ow_phys_closed_tx.py',
    'libero/phys/phys_closed_lingbot.py': 'protocol/runners/lingbot/phys_closed_lingbot.py',
    'libero/phys/phys_closed_openwam.py': 'protocol/runners/openwam/phys_closed_openwam.py',
    'libero/phys/phys_perturb.py': 'perturbations/phys/phys_perturb.py',
    'libero/phys/phys_truth.py': 'protocol/runners/lingbot/phys_truth.py',
    'libero/phys/t2_egn_closed.py': 'protocol/runners/lingbot/t2_egn_closed.py',
    'libero/phys/t2_k_closed.py': 'protocol/runners/lingbot/t2_k_closed.py',
    'libero/phys/t2_phys_truth.py': 'protocol/runners/lingbot/t2_phys_truth.py',
    'libero/recal_predicate.py': 'predicates/lingbot/recal_predicate.py',
    'libero/recal_v3.py': 'predicates/lingbot/recal_v3.py',
    'libero/recal_v4_t3.py': 'predicates/lingbot/recal_v4_t3.py',
    'libero/recal_wv.py': 'predicates/worldvla/recal_wv.py',
    'libero/regen_check.py': 'protocol/core/regen_check.py',
    'libero/rerun/h100/determinism_compare.py': 'protocol/rerun/determinism_compare.py',
    'libero/rerun/h100/lb_rerun_h100.py': 'protocol/rerun/lb_rerun_h100.py',
    'libero/rerun/h100/merge_manifest.py': 'protocol/rerun/merge_manifest.py',
    'libero/rerun/h100/ow_rerun_h100.py': 'protocol/rerun/ow_rerun_h100.py',
    'libero/rerun/h100/smoke660_check.py': 'protocol/checks/smoke660_check.py',
    'libero/rerun/lb_rerun.py': 'protocol/rerun/lb_rerun.py',
    'libero/rerun/ow_lastcall.py': 'protocol/rerun/ow_lastcall.py',
    'libero/rerun/ow_rerun.py': 'protocol/rerun/ow_rerun.py',
    'libero/rerun/ow_x86_rerun.py': 'protocol/rerun/ow_x86_rerun.py',
    'libero/rerun/replay_frames.py': 'protocol/rerun/replay_frames.py',
    'libero/rescore_base.py': 'predicates/lingbot/rescore_base.py',
    'libero/rescore_t3_v4.py': 'predicates/lingbot/rescore_t3_v4.py',
    'libero/t2_replay.py': 'protocol/runners/lingbot/t2_replay.py',
    'libero/t3_labels.py': 'predicates/lingbot/t3_labels.py',
    'libero/t3_subgoal_diag.py': 'predicates/rules/t3_subgoal_diag.py',
    'libero/t5_failure_modes.py': 'analysis/cells/t5_failure_modes.py',
    'libero/t5_geometry.py': 'predicates/rules/t5_geometry.py',
    'libero/t5_release_check.py': 'predicates/rules/t5_release_check.py',
    'libero/traj_rule_v2.py': 'predicates/rules/traj_rule_v2.py',
    'libero/worldvla/worldvla_unit.py': 'protocol/runners/worldvla/worldvla_unit.py',
    'libero/wv_content.py': 'analysis/cells/wv_content.py',
    'libero/wv_e4_analysis.py': 'analysis/cells/wv_e4_analysis.py',
    'libero/wv_genrow.py': 'predicates/worldvla/wv_genrow.py',
    'libero/wv_roundtrip.py': 'predicates/worldvla/wv_roundtrip.py',
    'libero/wv_train.py': 'predicates/worldvla/wv_train.py',
    'libero_env_adapter.py': 'protocol/core/libero_env_adapter.py',
    'verify_run.py': 'protocol/core/verify_run.py',
}
_SKIP = {"third_party", "__pycache__", "docs", "env", "patches", ".git"}


def P(s):
    strict = os.environ.get("WMV_STRICT_PATHS") == "1"

    def sub(m):
        v = os.environ.get(m.group(2))
        if v is None:
            if strict:
                raise KeyError(f"environment variable {m.group(2)} is not set (path {s!r}); see README, section Paths")
            return m.group(0)
        return v
    return _RX.sub(sub, s)


def code(legacy_rel):
    return os.path.join(REPO, LEGACY_CODE[legacy_rel])


def _code_dirs():
    out = []
    for d, ds, fs in os.walk(REPO):
        ds[:] = sorted(x for x in ds if x not in _SKIP)
        if any(f.endswith(".py") for f in fs):
            out.append(d)
    return out


CODE_DIRS = _code_dirs()


def bootstrap():
    for d in reversed(CODE_DIRS):
        if d not in sys.path:
            sys.path.insert(0, d)
