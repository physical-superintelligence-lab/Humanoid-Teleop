import os
import re
from pathlib import Path


class ProgressTracker:
    def __init__(self, base_dir='data', debug=False):
        self.base_dir = base_dir
        self.debug = debug
        self.total_stats = {}
        self.category_details = {}
        self.total_episodes_per_task = 40  # Expected number of episodes
        
    def _log(self, message):
        if self.debug:
            print(message)
            
    def scan_directory(self):
        # Get all categories (top-level directories)
        categories = [d for d in os.listdir(self.base_dir) 
                    if os.path.isdir(os.path.join(self.base_dir, d)) and not d.startswith('.')]
        
        self._log(f"Found categories: {categories}")
        
        # Initialize statistics
        self.total_stats = {
            'total_categories': len(categories),
            'completed_categories': 0,
            'total_subcategories': 0,
            'completed_subcategories': 0,
            'total_tasks': 0,
            'completed_tasks': 0
        }
        
        # Clear category details
        self.category_details = {}
        
        # Process each category
        for category in categories:
            self._process_category(category)
            
        # Calculate overall episode progress
        self._calculate_overall_episode_progress()
        
        return self.total_stats, self.category_details
        
    def _process_category(self, category):
        """Process a single category and update statistics."""
        category_path = os.path.join(self.base_dir, category)
        
        # Get all subcategories within this category
        subcategories = [d for d in os.listdir(category_path) 
                        if os.path.isdir(os.path.join(category_path, d))]
        
        self._log(f"\nCategory: {category}")
        self._log(f"Found subcategories: {subcategories}")
        
        # Initialize category stats
        category_stats = {
            'total_subcategories': len(subcategories),
            'completed_subcategories': 0,
            'total_tasks': 0,
            'completed_tasks': 0
        }
        
        # Store subcategory details
        subcategory_details = {}
        
        # Process each subcategory
        for subcategory in subcategories:
            self._process_subcategory(category, subcategory, category_stats, subcategory_details)
        
        # A category is considered complete if all subcategories are complete
        is_category_complete = (category_stats['completed_subcategories'] == category_stats['total_subcategories'] and 
                               category_stats['total_subcategories'] > 0)
        
        # Store category details
        self.category_details[category] = {
            'subcategories': category_stats['total_subcategories'],
            'completed_subcategories': category_stats['completed_subcategories'],
            'tasks': category_stats['total_tasks'],
            'completed_tasks': category_stats['completed_tasks'],
            'progress': category_stats['completed_subcategories'] / category_stats['total_subcategories'] * 100 if category_stats['total_subcategories'] > 0 else 0,
            'complete': is_category_complete,
            'subcategory_details': subcategory_details
        }
        
        # Update total stats
        self.total_stats['total_subcategories'] += category_stats['total_subcategories']
        self.total_stats['completed_subcategories'] += category_stats['completed_subcategories']
        self.total_stats['total_tasks'] += category_stats['total_tasks']
        self.total_stats['completed_tasks'] += category_stats['completed_tasks']
        if is_category_complete:
            self.total_stats['completed_categories'] += 1
            
    def _process_subcategory(self, category, subcategory, category_stats, subcategory_details):
        """Process a single subcategory and update statistics."""
        category_path = os.path.join(self.base_dir, category)
        subcategory_path = os.path.join(category_path, subcategory)
        
        # Get all tasks within this subcategory
        tasks = [d for d in os.listdir(subcategory_path) 
                if os.path.isdir(os.path.join(subcategory_path, d))]
        
        self._log(f"  Subcategory: {subcategory}")
        self._log(f"  Found tasks: {tasks}")
        
        # Initialize subcategory stats
        subcategory_stats = {
            'total_tasks': len(tasks),
            'completed_tasks': 0
        }
        
        # Store task details
        task_details = {}
        
        # Process each task
        for task in tasks:
            self._process_task(category, subcategory, task, subcategory_stats, task_details)
        
        # A subcategory is considered complete if all tasks are complete
        is_subcategory_complete = (subcategory_stats['completed_tasks'] == subcategory_stats['total_tasks'] and 
                                   subcategory_stats['total_tasks'] > 0)
        
        # Store subcategory details
        subcategory_details[subcategory] = {
            'tasks': subcategory_stats['total_tasks'],
            'completed_tasks': subcategory_stats['completed_tasks'],
            'progress': subcategory_stats['completed_tasks'] / subcategory_stats['total_tasks'] * 100 if subcategory_stats['total_tasks'] > 0 else 0,
            'complete': is_subcategory_complete,
            'task_details': task_details
        }
        
        # Update category stats
        category_stats['total_tasks'] += subcategory_stats['total_tasks']
        category_stats['completed_tasks'] += subcategory_stats['completed_tasks']
        if is_subcategory_complete:
            category_stats['completed_subcategories'] += 1
            
    def _process_task(self, category, subcategory, task, subcategory_stats, task_details):
        """Process a single task and update statistics."""
        category_path = os.path.join(self.base_dir, category)
        subcategory_path = os.path.join(category_path, subcategory)
        task_path = os.path.join(subcategory_path, task)
        
        # Get all episode directories
        try:
            episode_dirs = [d for d in os.listdir(task_path) 
                            if os.path.isdir(os.path.join(task_path, d)) and d.startswith('episode_')]
            
            self._log(f"    Task: {task}")
            self._log(f"    Found episode directories: {len(episode_dirs)}")
        except Exception as e:
            self._log(f"    Error listing episodes for task {task}: {str(e)}")
            episode_dirs = []
        
        # Count only episodes that contain data.json
        completed_episodes = 0
        for episode in episode_dirs:
            episode_path = os.path.join(task_path, episode)
            data_json_path = os.path.join(episode_path, 'data.json')
            
            self._log(f"      Checking {data_json_path}")
            self._log(f"      Exists: {os.path.exists(data_json_path)}")
            self._log(f"      Is file: {os.path.isfile(data_json_path) if os.path.exists(data_json_path) else 'N/A'}")
            
            if os.path.exists(data_json_path) and os.path.isfile(data_json_path):
                completed_episodes += 1
                self._log(f"      ✓ Counted episode {episode}")
            else:
                self._log(f"      ✗ Episode {episode} does not have data.json")
                # Delete incomplete episode directory
                try:
                    import shutil
                    shutil.rmtree(episode_path)
                    self._log(f"      Deleted incomplete episode directory: {episode_path}")
                except Exception as e:
                    self._log(f"      Error deleting directory {episode_path}: {str(e)}")
        
        self._log(f"    Total completed episodes with data.json: {completed_episodes}")
        
        # A task is considered complete if all episodes are present with data.json
        is_complete = (completed_episodes >= self.total_episodes_per_task)
        
        # Store task details
        task_details[task] = {
            'path': task_path,
            'episodes': f"{completed_episodes}/{self.total_episodes_per_task}",
            'progress': completed_episodes / self.total_episodes_per_task * 100,
            'complete': is_complete,
            'completed_episodes': completed_episodes  # Store this for later use
        }
        
        # Update subcategory stats if task is complete
        if is_complete:
            subcategory_stats['completed_tasks'] += 1
            
    def _calculate_overall_episode_progress(self):
        """Calculate the overall episode progress across all tasks."""
        total_episodes_needed = self.total_stats['total_tasks'] * self.total_episodes_per_task
        total_episodes_completed = 0
        
        for category_stats in self.category_details.values():
            for subcategory_stats in category_stats['subcategory_details'].values():
                for task_stats in subcategory_stats['task_details'].values():
                    total_episodes_completed += task_stats['completed_episodes']
        
        overall_progress = (total_episodes_completed / total_episodes_needed * 100) if total_episodes_needed > 0 else 0
        
        self.total_stats['total_episodes'] = total_episodes_needed
        self.total_stats['completed_episodes'] = total_episodes_completed
        self.total_stats['overall_progress'] = overall_progress

    def get_finished(self, task_path):
        directory = Path(task_path)
        if not directory.is_dir():
            return 0

        episode_pattern = re.compile(r"episode_(\d+)$")
        episodes = {}

        # Gather all episodes and check for data.json
        for item in directory.iterdir():
            if item.is_dir():
                match = episode_pattern.match(item.name)
                if match:
                    episode_num = int(match.group(1))
                    data_json_path = item / 'data.json'
                    
                    # Check if the episode has data.json
                    if data_json_path.is_file():
                        episodes[episode_num] = True
                    else:
                        # Delete incomplete episode directory
                        import shutil
                        try:
                            shutil.rmtree(item)
                            self._log(f"Deleted incomplete episode directory: {item}")
                        except Exception as e:
                            self._log(f"Error deleting directory {item}: {str(e)}")

        # Find the last consecutive episode index
        if not episodes:
            return 0
            
        episode_numbers = sorted(episodes.keys())
        expected = 0
        for num in episode_numbers:
            if num != expected:
                break
            expected += 1

        self._log(f"Next consecutive episode index is {expected}")
        return expected
        
    def get_next(self):
        """
        Get the path of the next unfinished task with the most progress,
        including the next episode directory path.
        
        Returns:
            str: Path to the next episode directory to work on, or None if all tasks are complete
        """
        self.scan_directory()
        unfinished_tasks = []
        
        for category, cat_stats in self.category_details.items():
            for subcategory, sub_stats in cat_stats['subcategory_details'].items():
                for task, task_stats in sub_stats['task_details'].items():
                    if not task_stats['complete']:
                        # Get the next episode index for this task
                        next_episode = self.get_finished(task_stats['path'])
                        
                        unfinished_tasks.append({
                            'category': category,
                            'subcategory': subcategory,
                            'task': task,
                            'path': task_stats['path'],
                            'progress': task_stats['progress'],
                            'completed_episodes': task_stats['completed_episodes'],
                            'next_episode': next_episode
                        })
        
        if not unfinished_tasks:
            return None
        
        # Sort by progress (highest first)
        unfinished_tasks.sort(key=lambda x: x['progress'], reverse=True)
        
        # Get the task with most progress
        next_task = unfinished_tasks[0]
        
        # Build the path to the next episode directory
        next_episode_path = os.path.join(next_task['path'], f"episode_{next_task['next_episode']}")
        
        # Create the directory if it doesn't exist
        os.makedirs(next_episode_path, exist_ok=True)
        
        return next_episode_path
    
    def display_progress(self):
        """
        Display the progress of completed episodes, tasks, and subcategories.
        """
        # Make sure we have the latest data
        if not self.total_stats:
            self.scan_directory()
        
        # Display function for progress bars
        def create_progress_bar(progress, width=40):
            filled_length = int(width * progress / 100)
            bar = '█' * filled_length + '░' * (width - filled_length)
            return f"[{bar}] {progress:.1f}%"
        
        # Find the longest task name for alignment
        max_task_name_length = 0
        for category_stats in self.category_details.values():
            for subcategory_stats in category_stats['subcategory_details'].values():
                for task in subcategory_stats['task_details'].keys():
                    task_name = task.replace('_', ' ').title()
                    max_task_name_length = max(max_task_name_length, len(task_name))
        
        # Display the progress
        print("\n" + "=" * 80)
        print(f"OVERALL PROGRESS SUMMARY")
        print("=" * 80)
        print(f"Categories: {self.total_stats['completed_categories']}/{self.total_stats['total_categories']} complete")
        print(f"Subcategories: {self.total_stats['completed_subcategories']}/{self.total_stats['total_subcategories']} complete")
        print(f"Tasks: {self.total_stats['completed_tasks']}/{self.total_stats['total_tasks']} complete")
        print(f"Overall Episode Progress: {self.total_stats['completed_episodes']}/{self.total_stats['total_episodes']} episodes ({self.total_stats['overall_progress']:.1f}%)")
        print(f"Progress Bar: {create_progress_bar(self.total_stats['overall_progress'])}")
        print("\n" + "=" * 80)
        
        # Display detailed progress for each category
        for category, stats in self.category_details.items():
            print(f"\nCATEGORY: {category.replace('_', ' ').title()}")
            print("-" * 80)
            print(f"Subcategories: {stats['completed_subcategories']}/{stats['subcategories']} complete")
            print(f"Tasks: {stats['completed_tasks']}/{stats['tasks']} complete")
            print(f"Progress: {create_progress_bar(stats['progress'])}")
            
            # Display detailed progress for each subcategory
            for subcategory, sub_stats in sorted(stats['subcategory_details'].items()):
                print(f"\n  SUBCATEGORY: {subcategory.replace('_', ' ').title()}")
                print(f"  Tasks: {sub_stats['completed_tasks']}/{sub_stats['tasks']} complete")
                print(f"  Progress: {create_progress_bar(sub_stats['progress'], width=30)}")
                
                # Calculate subcategory episode progress
                total_subcategory_episodes = sub_stats['tasks'] * self.total_episodes_per_task
                completed_subcategory_episodes = sum(task_stats['completed_episodes'] for task_stats in sub_stats['task_details'].values())
                subcategory_episode_progress = (completed_subcategory_episodes / total_subcategory_episodes * 100) if total_subcategory_episodes > 0 else 0
                print(f"  Episode Progress: {completed_subcategory_episodes}/{total_subcategory_episodes} episodes ({subcategory_episode_progress:.1f}%)")
                
                # Display detailed progress for each task with aligned progress bars
                for task, task_stats in sorted(sub_stats['task_details'].items()):
                    task_name = task.replace('_', ' ').title()
                    status = "✓" if task_stats['complete'] else " "
                    # Format with fixed width for task name to align progress bars
                    print(f"    [{status}] {task_name:{max_task_name_length}} : {task_stats['episodes']} episodes - {create_progress_bar(task_stats['progress'], width=20)}")
        
        print("\n" + "=" * 80)
        
        # List all tasks with at least one episode completed
        self.display_tasks_with_progress()
        
    def display_tasks_with_progress(self):
        """Display a list of all tasks with at least one episode completed."""
        tasks_with_progress = []
        for category, cat_stats in self.category_details.items():
            for subcategory, sub_stats in cat_stats['subcategory_details'].items():
                for task, task_stats in sub_stats['task_details'].items():
                    if task_stats['completed_episodes'] > 0:
                        tasks_with_progress.append((
                            category,
                            subcategory,
                            task,
                            task_stats['completed_episodes'],
                            task_stats['progress']
                        ))
        
        if tasks_with_progress:
            print("\nTASKS WITH PROGRESS:")
            print("-" * 80)
            tasks_with_progress.sort(key=lambda x: x[4], reverse=True)  # Sort by completion percentage
            
            for category, subcategory, task, episodes, progress in tasks_with_progress:
                cat_name = category.replace('_', ' ').title()
                subcat_name = subcategory.replace('_', ' ').title()
                task_name = task.replace('_', ' ').title()
                print(f"{cat_name} > {subcat_name} > {task_name}: {episodes}/{self.total_episodes_per_task} episodes ({progress:.1f}%)")
        
        print("\n" + "=" * 80)
        
        # Display the next unfinished task
        next_task = self.get_next()
        if next_task:
            print(f"\nNEXT UNFINISHED TASK (with most progress):")
            print(f"{next_task}")
            print("=" * 80)
        else:
            print("\nAll tasks are complete!")
            print("=" * 80)


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Display progress of task completion')
    parser.add_argument('-d', '--directory', default='data', help='Base directory containing category folders')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--next', action='store_true', help='Only display the next unfinished task')
    
    args = parser.parse_args()
    
    tracker = ProgressTracker(args.directory, args.debug)
    
    if args.next:
        next_task = tracker.get_next()
        if next_task:
            print(f"Next unfinished task: {next_task}")
        else:
            print("All tasks are complete!")
    else:
        tracker.display_progress()
