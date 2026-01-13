#!/usr/bin/env python3
"""
Main Pipeline Runner Script
Chạy toàn bộ pipeline (bước 1-7) cho một project cụ thể

Usage:
    python run_pipeline.py --project-key ZOOKEEPER
    python run_pipeline.py --project-key HBASE
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Get the scripts directory
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
TOOLS_DIR = PROJECT_ROOT / "tools"

# Define the pipeline steps
PIPELINE_STEPS = [
    {
        "step": 1,
        "script": "01_group_tasks.py",
        "args": ["--project_key"],
        "description": "Gom issues thành logical tasks"
    },
    {
        "step": 2,
        "script": "02_tag_logical_tasks.py",
        "args": ["--project-key"],
        "description": "Gán tags cho logical tasks"
    },
    {
        "step": 3,
        "script": "03_build_issue_dag.py",
        "args": ["--project-key"],
        "description": "Xây dựng DAG cho issues"
    },
    {
        "step": 4,
        "script": "04_build_logical_task_dag.py",
        "args": ["--project-key"],
        "description": "Xây dựng DAG cho logical tasks"
    },
    {
        "step": 5,
        "script": "05_topo_sort_logical_tasks.py",
        "args": ["--project-key"],
        "description": "Topo sort + tính duration"
    },
    {
        "step": 6,
        "script": "06_export_assignee_profiles.py",
        "args": ["--project-key"],
        "description": "Xuất assignee profiles"
    },
    {
        "step": 6.5,
        "script": "06b_assign_cost_to_assignees.py",
        "args": [],
        "description": "Gán cost cho assignees dựa trên skill profile"
    },
    {
        "step": 7,
        "script": "07_hs_topo_assign.py",
        "args": ["--project-key"],
        "description": "Gán công việc (HS)"
    },
    {
        "step": 7,
        "script": "07_ihs_topo_assign.py",
        "args": ["--project-key"],
        "description": "Gán công việc (IHS)"
    },
    {
        "step": 7,
        "script": "07_ghs_topo_assign.py",
        "args": ["--project-key"],
        "description": "Gán công việc (GHS)"
    },
    {
        "step": 7,
        "script": "07_mohs_topo_assign.py",
        "args": ["--project-key"],
        "description": "Gán công việc (MOHS)"
    },
]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Chạy pipeline xử lý JIRA cho một project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run_pipeline.py --project-key ZOOKEEPER
  python run_pipeline.py --project-key HBASE --skip-mohs
        """
    )
    
    parser.add_argument(
        "--project-key",
        type=str,
        default="ZOOKEEPER",
        help="Tên project (mặc định: ZOOKEEPER)"
    )
    
    parser.add_argument(
        "--with-step0",
        action="store_true",
        help="Chạy bước 0 (chuẩn bị toàn cục) trước bước 1-7"
    )
    
    parser.add_argument(
        "--skip-mohs",
        action="store_true",
        help="Bỏ qua bước MOHS (chỉ chạy HS/IHS/GHS)"
    )
    
    parser.add_argument(
        "--only-assignment",
        action="store_true",
        help="Chỉ chạy bước 7 (gán công việc) - bước 1-6 phải đã chạy"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In chi tiết thông tin chạy"
    )
    
    return parser.parse_args()


def run_script(script_path, project_key, args_flags, step_num, description, verbose=False):
    """
    Chạy một script với project key
    
    Args:
        script_path: Đường dẫn đến script
        project_key: Tên project
        args_flags: Danh sách flag arguments
        step_num: Số bước (cho hiển thị)
        description: Mô tả bước
        verbose: In chi tiết hay không
    
    Returns:
        True nếu thành công, False nếu lỗi
    """
    # Xây dựng command
    cmd = [sys.executable, str(script_path)]
    
    # Thêm project-key argument cho scripts có nó (01, 02, 03 có --project-key)
    script_name = script_path.name
    if script_name in ["01_group_tasks.py", "02_tag_logical_tasks.py", "03_build_issue_dag.py"]:
        for flag in args_flags:
            cmd.extend([flag, project_key])
    
    # Thêm output paths cho các scripts cụ thể
    script_name = script_path.name
    
    if script_name == "01_group_tasks.py":
        cmd.extend([
            "--tasks_out", f"projects/{project_key}/logical_tasks.csv",
            "--mapping_out", f"projects/{project_key}/issue_to_task_mapping.csv"
        ])
    elif script_name == "02_tag_logical_tasks.py":
        cmd.extend([
            "--tasks", f"projects/{project_key}/logical_tasks.csv",
            "--mapping", f"projects/{project_key}/issue_to_task_mapping.csv",
            "--output", f"projects/{project_key}/logical_tasks_tagged.csv"
        ])
    elif script_name == "03_build_issue_dag.py":
        cmd.extend([
            "--links", f"projects/{project_key}/issue_links.csv",
            "--out-nodes", f"projects/{project_key}/issue_dag_nodes.csv",
            "--out-edges", f"projects/{project_key}/issue_dag_edges.csv"
        ])
    elif script_name == "04_build_logical_task_dag.py":
        cmd.extend([
            "--tasks", f"projects/{project_key}/logical_tasks_tagged.csv",
            "--edges", f"projects/{project_key}/issue_dag_edges.csv",
            "--out-nodes", f"projects/{project_key}/logical_dag_nodes.csv",
            "--out-edges", f"projects/{project_key}/logical_dag_edges.csv"
        ])
    elif script_name == "05_topo_sort_logical_tasks.py":
        cmd.extend([
            "--nodes", f"projects/{project_key}/logical_dag_nodes.csv",
            "--edges", f"projects/{project_key}/logical_dag_edges.csv",
            "--output", f"projects/{project_key}/logical_topo.csv"
        ])
    elif script_name == "06_export_assignee_profiles.py":
        cmd.extend([
            "--output", f"projects/{project_key}/assignees.csv"
        ])
    elif script_name == "06b_assign_cost_to_assignees.py":
        # Script này không cần arguments - nó tự động đọc từ data/interim/assignee_skill_profile.csv
        # và xuất ra data/interim/assignee_cost_profile.csv
        pass
    elif script_name == "07_hs_topo_assign.py":
        cmd.extend([
            "--topo", f"projects/{project_key}/logical_topo.csv",
            "--assignees", f"projects/{project_key}/assignees.csv",
            "--output-assignment", f"projects/{project_key}/hs_assignment.csv",
            "--output-score", f"projects/{project_key}/hs_score.json",
            "--plot-dir", f"projects/{project_key}/hs_plots"
        ])
    elif script_name == "07_ihs_topo_assign.py":
        cmd.extend([
            "--topo", f"projects/{project_key}/logical_topo.csv",
            "--assignees", f"projects/{project_key}/assignees.csv",
            "--output-assignment", f"projects/{project_key}/ihs_assignment.csv",
            "--output-score", f"projects/{project_key}/ihs_score.json",
            "--plot-dir", f"projects/{project_key}/ihs_plots"
        ])
    elif script_name == "07_ghs_topo_assign.py":
        cmd.extend([
            "--topo", f"projects/{project_key}/logical_topo.csv",
            "--assignees", f"projects/{project_key}/assignees.csv",
            "--output-assignment", f"projects/{project_key}/ghs_assignment.csv",
            "--output-score", f"projects/{project_key}/ghs_score.json",
            "--plot-dir", f"projects/{project_key}/ghs_plots"
        ])
    elif script_name == "07_mohs_topo_assign.py":
        cmd.extend([
            "--topo", f"projects/{project_key}/logical_topo.csv",
            "--assignees", f"projects/{project_key}/assignees.csv",
            "--output-assignment", f"projects/{project_key}/mohs_assignment.csv",
            "--output-score", f"projects/{project_key}/mohs_score.json"
        ])
    
    print(f"\n{'='*70}")
    print(f"[BƯỚC {step_num}] {description}")
    print(f"{'='*70}")
    print(f"📝 Script: {script_path.name}")
    print(f"🎯 Project: {project_key}")
    
    if verbose:
        print(f"📋 Command: {' '.join(cmd)}")
    
    try:
        # Chạy script từ PROJECT_ROOT thay vì SCRIPTS_DIR để paths relative hoạt động đúng
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"✅ Bước {step_num} THÀNH CÔNG")
            return True
        else:
            print(f"❌ Bước {step_num} THẤT BẠI (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy bước {step_num}: {e}")
        return False


def validate_project_structure(project_key):
    """
    Kiểm tra cấu trúc thư mục project tồn tại
    
    Args:
        project_key: Tên project
    
    Returns:
        True nếu cấu trúc hợp lệ
    """
    project_dir = PROJECT_ROOT / "projects" / project_key
    
    if not project_dir.exists():
        print(f"⚠️  Thư mục project chưa tồn tại: {project_dir}")
        print(f"📁 Tạo thư mục...")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo các thư mục con cần thiết
        (project_dir / "hs_plots").mkdir(exist_ok=True)
        (project_dir / "ihs_plots").mkdir(exist_ok=True)
        (project_dir / "ghs_plots").mkdir(exist_ok=True)
        (project_dir / "mohs_plots").mkdir(exist_ok=True)
        
        print(f"✅ Thư mục đã tạo")
    
    return True


def check_global_data():
    """
    Kiểm tra xem dữ liệu toàn cục (step 0) đã được tạo hay chưa
    
    Returns:
        True nếu tất cả file cần thiết tồn tại
    """
    required_files = [
        PROJECT_ROOT / "data/raw/all_issues.csv",
        PROJECT_ROOT / "data/interim/all_issues_tagged.csv",
        PROJECT_ROOT / "data/interim/assignee_mapping.csv",
        PROJECT_ROOT / "data/interim/assignee_skill_profile.csv",
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    
    return len(missing_files) == 0


def check_project_issue_links(project_key):
    """
    Kiểm tra xem file issue_links.csv tồn tại cho project
    
    Returns:
        True nếu tồn tại, False nếu không
    """
    links_file = PROJECT_ROOT / "projects" / project_key / "issue_links.csv"
    return links_file.exists()


def check_cost_profile():
    """
    Check if assignee cost profile exists.
    """
    cost_file = PROJECT_ROOT / "data/interim/assignee_cost_profile.csv"
    return cost_file.exists()


def run_pipeline(args):
    """Run pipeline with parsed args"""
    project_key = args.project_key.upper()
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║                    JIRA PIPELINE RUNNER                           ║
║          Gán công việc tối ưu (HS/IHS/GHS/MOHS)                   ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"📋 Cấu hình:")
    print(f"  Project: {project_key}")
    print(f"  With Step 0: {args.with_step0}")
    print(f"  Skip MOHS: {args.skip_mohs}")
    print(f"  Only Assignment: {args.only_assignment}")
    print(f"  Verbose: {args.verbose}")
    
    # Kiểm tra dữ liệu toàn cục (step 0)
    print(f"\n🔍 Kiểm tra dữ liệu toàn cục (Step 0)...")
    global_data_exists = check_global_data()
    
    if not global_data_exists and not args.with_step0:
        print(f"\n⚠️  Dữ liệu toàn cục chưa được tạo!")
        print(f"📌 Cách khắc phục:")
        print(f"   • Chạy: python scripts/run_pipeline.py --project-key {project_key} --with-step0")
        print(f"   • Hoặc chạy riêng: python scripts/00_all_projects_assignee_skills.py")
        return 1
    elif global_data_exists:
        print(f"✅ Dữ liệu toàn cục đã tồn tại")
    
    # Kiểm tra file issue_links.csv cho project
    print(f"\n🔍 Kiểm tra issue_links.csv cho project {project_key}...")
    issue_links_exists = check_project_issue_links(project_key)
    
    if not issue_links_exists:
        print(f"\n⚠️  File issue_links.csv chưa được tạo cho project {project_key}!")
        print(f"📌 Cách khắc phục:")
        print(f"   • Chạy: python scripts/mongodata3.py")
        print(f"   • Sửa PROJECT_KEY trong script nếu cần")
        print(f"   • File sẽ được tạo tại: projects/{project_key}/issue_links.csv")
        return 1
    else:
        print(f"✅ issue_links.csv đã tồn tại")
    
    # Kiểm tra cấu trúc project
    print(f"\n🔍 Kiểm tra cấu trúc thư mục project...")
    validate_project_structure(project_key)
    
    # Chuẩn bị pipeline steps
    steps = []
    
    # Thêm step 0 nếu cần
    if args.with_step0:
        steps.append({
            "step": 0,
            "script": "00_all_projects_assignee_skills.py",
            "args": [],
            "description": "Chuẩn bị dữ liệu toàn cục (export issues + skill profiles)"
        })
    
    # Thêm steps 1-7
    steps.extend(PIPELINE_STEPS)
    
    # Lọc pipeline steps nếu cần
    if args.skip_mohs:
        steps = [s for s in steps if s["script"] != "07_mohs_topo_assign.py"]
    
    if args.only_assignment:
        steps = [s for s in steps if s["step"] == 7]
    
    # Chạy các bước
    failed_steps = []
    successful_steps = []
    
    print(f"\n🚀 Bắt đầu chạy pipeline ({len(steps)} bước)...\n")
    
    for step_info in steps:
        if step_info["script"] in {
            "07_hs_topo_assign.py",
            "07_ihs_topo_assign.py",
            "07_ghs_topo_assign.py",
        }:
            if not check_cost_profile():
                print(
                    "\nCost profile is required for HS/IHS/GHS. "
                    "Run step 6b to generate it: "
                    "python scripts/06b_assign_cost_to_assignees.py"
                )
                return 1

        script_path = SCRIPTS_DIR / step_info["script"]
        
        if not script_path.exists():
            print(f"❌ Script không tồn tại: {script_path}")
            failed_steps.append(step_info["script"])
            continue
        
        success = run_script(
            script_path,
            project_key,
            step_info["args"],
            step_info["step"],
            step_info["description"],
            verbose=args.verbose
        )
        
        if success:
            successful_steps.append(step_info["script"])
        else:
            failed_steps.append(step_info["script"])
            # Hỏi có muốn tiếp tục không
            response = input(f"\n⚠️  Bước {step_info['step']} thất bại. Tiếp tục? (y/n): ").strip().lower()
            if response != "y":
                print("❌ Dừng pipeline")
                break
    
    # Tóm tắt kết quả
    print(f"\n{'='*70}")
    print(f"📊 TÓM TẮT KẾT QUẢ")
    print(f"{'='*70}")
    print(f"✅ Thành công: {len(successful_steps)} bước")
    print(f"❌ Thất bại: {len(failed_steps)} bước")
    
    if successful_steps:
        print(f"\nBước thành công:")
        for script in successful_steps:
            print(f"  ✅ {script}")
    
    if failed_steps:
        print(f"\nBước thất bại:")
        for script in failed_steps:
            print(f"  ❌ {script}")
    
    # Output location
    project_output = PROJECT_ROOT / "projects" / project_key
    print(f"\n📁 Output location: {project_output}")
    
    if len(failed_steps) == 0:
        print(f"\n🎉 Pipeline hoàn tất thành công cho project {project_key}!")
        print(f"\n📊 Xem biểu đồ kết quả:")
        print(f"  python scripts/visualize_mohs.py")
        return 0
    else:
        print(f"\n⚠️  Pipeline hoàn tất với {len(failed_steps)} lỗi")
        return 1




def run_tool_script(script_name, args_list=None):
    args_list = args_list or []
    script_path = TOOLS_DIR / script_name
    if not script_path.exists():
        print(f"Tool not found: {script_path}")
        return 1

    cmd = [sys.executable, str(script_path)] + args_list
    print("\n" + "=" * 70)
    print(f"[Tool] {script_name}")
    print("=" * 70)
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=False,
        )
        return result.returncode
    except Exception as e:
        print(f"Error running tool {script_name}: {e}")
        return 1





def _prompt_int(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Invalid number. Using default.")
        return default


def _prompt_bool(prompt, default=False):
    while True:
        suffix = "Y/n" if default else "y/N"
        raw = input(f"{prompt} ({suffix}): ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please enter y or n.")


def _prompt_text(prompt, default=None):
    raw = input(f"{prompt}{' [' + default + ']' if default else ''}: ").strip()
    return raw or (default or "")


def interactive_menu():
    while True:
        print("\n=== JIRA PIPELINE MENU ===")
        print("1) Run pipeline (Step 1-7)")
        print("2) Run pipeline with Step 0")
        print("3) Run only assignment (Step 7)")
        print("4) Run pipeline without MOHS")
        print("5) Run with verbose logs")
        print("6) Tool: Compare algorithms")
        print("7) Tool: Render Gantt from assignment")
        print("8) Tool: Visualize MOHS (Pareto plots)")
        print("9) Tool: Legacy Gantt (uses issue links)")
        print("0) Exit")

        choice = input("Choose an option: ").strip()
        if choice == "0":
            return 0

        if choice in {"6", "7", "8", "9"}:
            project_key = _prompt_text("Project key", "ZOOKEEPER").upper()
            if choice == "6":
                out_csv = _prompt_text("Output CSV (optional)", "").strip()
                args = ["--project", project_key]
                if out_csv:
                    args += ["--out", out_csv]
                return run_tool_script("compare_algorithms.py", args)

            if choice == "7":
                default_assign = f"projects/{project_key}/ihs_assignment.csv"
                assignment = _prompt_text("Assignment CSV", default_assign)
                output = _prompt_text("Output file (optional)", "")
                max_tasks = _prompt_int("Max tasks (0 = all)", 0)
                args = ["--assignment", assignment]
                if output:
                    args += ["--output", output]
                if max_tasks and max_tasks > 0:
                    args += ["--max-tasks", str(max_tasks)]
                return run_tool_script("render_gantt_from_assignment.py", args)

            if choice == "8":
                score = _prompt_text("Score JSON", f"projects/{project_key}/mohs_score.json")
                out_dir = _prompt_text("Output dir", f"projects/{project_key}/mohs_plots")
                args = ["--score", score, "--out-dir", out_dir]
                return run_tool_script("visualize_mohs.py", args)

            if choice == "9":
                assignment = _prompt_text("Assignment CSV", f"projects/{project_key}/hs_assignment.csv")
                links = _prompt_text("Issue links CSV", f"projects/{project_key}/issue_links.csv")
                out_dir = _prompt_text("Output dir", f"projects/{project_key}/gantt_legacy")
                max_tasks = _prompt_int("Max tasks (default 200)", 200)
                args = ["--assignment", assignment, "--links", links, "--output-dir", out_dir]
                if max_tasks is not None:
                    args += ["--max-tasks", str(max_tasks)]
                return run_tool_script("visualize_gantt.py", args)

        project_key = _prompt_text("Project key", "ZOOKEEPER").upper()

        args = argparse.Namespace(
            project_key=project_key,
            with_step0=False,
            skip_mohs=False,
            only_assignment=False,
            verbose=False,
        )

        if choice == "1":
            pass
        elif choice == "2":
            args.with_step0 = True
        elif choice == "3":
            args.only_assignment = True
        elif choice == "4":
            args.skip_mohs = True
        elif choice == "5":
            args.verbose = True
        else:
            print("Invalid option. Try again.")
            continue

        if choice not in ("2", "3", "4", "5"):
            args.with_step0 = _prompt_bool("Include Step 0?", False)
            args.skip_mohs = _prompt_bool("Skip MOHS?", False)
            args.only_assignment = _prompt_bool("Only assignment (Step 7)?", False)
            args.verbose = _prompt_bool("Verbose logs?", False)

        return run_pipeline(args)



def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        return interactive_menu()

    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
