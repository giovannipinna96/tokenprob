#!/usr/bin/env python3
"""
Migration Script for Forced Visualizations

This script migrates the existing flat structure of forced_visualizations/
to the new hierarchical structure that matches model_visualizations/.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def migrate_forced_visualizations():
    """
    Migrate forced visualizations from flat to hierarchical structure.
    """
    print("🔄 MIGRATING FORCED VISUALIZATIONS TO HIERARCHICAL STRUCTURE")
    print("="*80)

    # Define paths
    old_dir = Path("forced_visualizations")
    backup_dir = Path("forced_visualizations_backup")
    new_base_dir = Path("forced_visualizations_organized")

    # Check if old directory exists
    if not old_dir.exists():
        print(f"❌ Source directory {old_dir} does not exist!")
        return False

    print(f"📁 Source directory: {old_dir}")
    print(f"📁 Backup directory: {backup_dir}")
    print(f"📁 New organized directory: {new_base_dir}")

    # Step 1: Create backup of existing files
    if old_dir.exists():
        print(f"\n📦 Creating backup...")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(old_dir, backup_dir)
        print(f"✅ Backup created at: {backup_dir}")

    # Step 2: Create new organized structure
    print(f"\n🏗️ Creating new hierarchical structure...")

    # Create base directory and model subdirectory
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model_safe_name = model_name.replace("/", "_").replace("-", "_").replace(":", "_")
    model_dir = new_base_dir / model_safe_name

    new_base_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    print(f"✅ Created: {new_base_dir}")
    print(f"✅ Created: {model_dir}")

    # Step 3: Define file mapping (old -> new)
    file_mappings = {
        # Individual visualizations
        "factorial_correct_logits.html": f"{model_dir}/factorial_correct_logits.html",
        "factorial_correct_probability.html": f"{model_dir}/factorial_correct_probability.html",
        "factorial_correct_rank.html": f"{model_dir}/factorial_correct_rank.html",
        "factorial_correct_surprisal.html": f"{model_dir}/factorial_correct_surprisal.html",

        "factorial_buggy_logits.html": f"{model_dir}/factorial_buggy_logits.html",
        "factorial_buggy_probability.html": f"{model_dir}/factorial_buggy_probability.html",
        "factorial_buggy_rank.html": f"{model_dir}/factorial_buggy_rank.html",
        "factorial_buggy_surprisal.html": f"{model_dir}/factorial_buggy_surprisal.html",

        # Comparison visualizations
        "comparison_logits.html": f"{model_dir}/comparison_logits.html",
        "comparison_probability.html": f"{model_dir}/comparison_probability.html",
        "comparison_rank.html": f"{model_dir}/comparison_rank.html",

        # Analysis report
        "detailed_analysis_report.html": f"{model_dir}/detailed_analysis_report.html"
    }

    # Step 4: Move files to new structure
    print(f"\n📋 Migrating {len(file_mappings)} files...")
    migrated_count = 0

    for old_file, new_file in file_mappings.items():
        old_path = old_dir / old_file
        new_path = Path(new_file)

        if old_path.exists():
            # Copy file to new location
            shutil.copy2(old_path, new_path)
            print(f"  ✅ {old_file} -> {new_path}")
            migrated_count += 1
        else:
            print(f"  ⚠️ {old_file} not found, skipping")

    print(f"📊 Migrated {migrated_count}/{len(file_mappings)} files")

    # Step 5: Generate placeholder JSON data files if they don't exist
    print(f"\n📝 Creating placeholder JSON data files...")

    json_files = [
        f"{model_dir}/factorial_correct_analysis.json",
        f"{model_dir}/factorial_buggy_analysis.json"
    ]

    placeholder_json = {
        "metadata": {
            "note": "Placeholder JSON - run new analysis to get actual data",
            "model_name": model_name,
            "migrated_at": datetime.now().isoformat()
        },
        "summary_statistics": {},
        "tokens": []
    }

    import json
    for json_file in json_files:
        json_path = Path(json_file)
        if not json_path.exists():
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(placeholder_json, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Created placeholder: {json_path}")

    # Step 6: Generate index pages using the visualizer
    print(f"\n📄 Generating index pages...")

    try:
        from forced_visualizer import ForcedGenerationVisualizer

        visualizer = ForcedGenerationVisualizer()

        # Generate model-specific index
        analysis_results = {
            "examples_count": 2,
            "visualizations_count": migrated_count,
            "processing_time": 0  # Unknown for migrated files
        }

        model_index = visualizer.generate_model_index(
            str(model_dir),
            model_name,
            analysis_results
        )
        print(f"  ✅ Created model index: {model_index}")

        # Generate main index
        models_tested = [{
            "model_name": model_name,
            "status": "success",
            "examples_count": 2,
            "visualizations_count": migrated_count,
            "processing_time": 0
        }]

        main_index = visualizer.generate_main_index(str(new_base_dir), models_tested)
        print(f"  ✅ Created main index: {main_index}")

    except Exception as e:
        print(f"  ⚠️ Could not generate index pages: {e}")
        print(f"    You can generate them later by running the new test script")

    # Step 7: Summary
    print(f"\n🎉 MIGRATION COMPLETED!")
    print(f"📊 Summary:")
    print(f"  • Migrated files: {migrated_count}")
    print(f"  • New structure: {new_base_dir}")
    print(f"  • Backup available: {backup_dir}")
    print(f"  • Model directory: {model_dir}")

    print(f"\n📁 New directory structure:")
    print(f"{new_base_dir}/")
    print(f"├── index.html")
    print(f"└── {model_safe_name}/")
    print(f"    ├── index.html")
    print(f"    ├── factorial_correct_*.html")
    print(f"    ├── factorial_buggy_*.html")
    print(f"    ├── comparison_*.html")
    print(f"    ├── detailed_analysis_report.html")
    print(f"    ├── factorial_correct_analysis.json")
    print(f"    └── factorial_buggy_analysis.json")

    print(f"\n🔗 To view results:")
    print(f"  Open: {new_base_dir}/index.html")

    return True


def cleanup_old_structure():
    """
    Optional: Remove the old flat structure after confirming migration worked.
    """
    old_dir = Path("forced_visualizations")

    if old_dir.exists():
        print(f"\n🗑️ CLEANUP: Remove old flat structure?")
        response = input(f"Delete {old_dir}? (y/N): ").lower().strip()

        if response == 'y':
            shutil.rmtree(old_dir)
            print(f"✅ Removed old directory: {old_dir}")
        else:
            print(f"💾 Keeping old directory: {old_dir}")


def main():
    """Main migration function."""
    try:
        success = migrate_forced_visualizations()

        if success:
            print(f"\n✅ Migration completed successfully!")

            # Ask if user wants to cleanup
            cleanup_old_structure()

            print(f"\n🚀 Next steps:")
            print(f"1. Verify the new structure looks correct")
            print(f"2. Run the new test script to generate fresh data:")
            print(f"   uv run python test_forced_visualization.py")
            print(f"3. Compare with model_visualizations/ structure")

        else:
            print(f"\n❌ Migration failed!")

    except Exception as e:
        print(f"\n💥 Migration error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()