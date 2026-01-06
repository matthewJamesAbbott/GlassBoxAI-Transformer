#!/usr/bin/env python3
"""
Transformer CLI GUI - Interactive Frontend for transformer.cu
Matthew Abbott 2025

A terminal-based GUI for the CUDA Transformer implementation.
Supports all transformer CLI arguments, file I/O, compilation, and chat.

Usage:
    python3 transformer_gui.py
    
Requirements:
    - Python 3.6+
    - nvcc (NVIDIA CUDA Compiler)
    - transformer.cu in same directory
"""

import os
import sys
import subprocess
import readline
import json
import shutil
from pathlib import Path
from datetime import datetime

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

# Configuration
CONFIG = {
    'transformer_src': 'transformer.cu',
    'transformer_bin': 'transformer',
    'default_model': None,
    'default_tokenizer': None,
    'history_file': os.path.expanduser('~/.transformer_gui_history'),
    'config_file': os.path.expanduser('~/.transformer_gui_config.json'),
    'compile_flags': '-O2 -std=c++11',
    'max_tokens': 50,
    'temperature': 0.8,
    'log_file': 'transformer_gui.log'
}

class TransformerGUI:
    def __init__(self):
        self.model_path = None
        self.tokenizer_path = None
        self.conversation_history = []
        self.last_output = None
        self.settings = {
            'max_tokens': CONFIG['max_tokens'],
            'temperature': CONFIG['temperature'],
            'top_k': -1,
            'top_p': 1.0,
            'repetition_penalty': 1.0,
            'context_length': 1024,
            'seed': None,
            'cpu_layers': None,
            'all_cpu': False,
            'verbose': False,
            'device': 0
        }
        self.load_config()
        self.setup_readline()
        
    def load_config(self):
        """Load saved configuration"""
        if os.path.exists(CONFIG['config_file']):
            try:
                with open(CONFIG['config_file'], 'r') as f:
                    saved = json.load(f)
                    if 'model_path' in saved:
                        self.model_path = saved['model_path']
                    if 'tokenizer_path' in saved:
                        self.tokenizer_path = saved['tokenizer_path']
                    if 'settings' in saved:
                        self.settings.update(saved['settings'])
            except:
                pass
                
    def save_config(self):
        """Save configuration"""
        try:
            with open(CONFIG['config_file'], 'w') as f:
                json.dump({
                    'model_path': self.model_path,
                    'tokenizer_path': self.tokenizer_path,
                    'settings': self.settings
                }, f, indent=2)
        except:
            pass
            
    def setup_readline(self):
        """Setup command history"""
        if os.path.exists(CONFIG['history_file']):
            try:
                readline.read_history_file(CONFIG['history_file'])
            except:
                pass
        readline.set_history_length(1000)
        
    def save_history(self):
        """Save command history"""
        try:
            readline.write_history_file(CONFIG['history_file'])
        except:
            pass

    def print_header(self):
        """Print welcome header"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}  TRANSFORMER CLI GUI - CUDA Implementation Frontend{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}  Interactive Chat, Compilation, and File Management{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}\n")
        
    def print_status(self):
        """Print current status"""
        print(f"{Colors.DIM}─────────────────────────────────────────{Colors.RESET}")
        
        # Binary status
        bin_exists = os.path.exists(CONFIG['transformer_bin'])
        bin_status = f"{Colors.GREEN}✓ compiled{Colors.RESET}" if bin_exists else f"{Colors.RED}✗ not compiled{Colors.RESET}"
        print(f"  Binary: {bin_status}")
        
        # Model status
        if self.model_path and os.path.exists(self.model_path):
            size_mb = os.path.getsize(self.model_path) / (1024*1024)
            print(f"  Model:  {Colors.GREEN}{os.path.basename(self.model_path)}{Colors.RESET} ({size_mb:.1f} MB)")
        else:
            print(f"  Model:  {Colors.YELLOW}not loaded{Colors.RESET}")
            
        # Tokenizer status
        if self.tokenizer_path and os.path.exists(self.tokenizer_path):
            print(f"  Tokenizer: {Colors.GREEN}{os.path.basename(self.tokenizer_path)}{Colors.RESET}")
        else:
            print(f"  Tokenizer: {Colors.YELLOW}not loaded{Colors.RESET}")
            
        # Settings
        device = "CPU" if self.settings['all_cpu'] else f"GPU:{self.settings['device']}"
        print(f"  Device: {Colors.BLUE}{device}{Colors.RESET} | Temp: {self.settings['temperature']} | MaxTok: {self.settings['max_tokens']}")
        print(f"{Colors.DIM}─────────────────────────────────────────{Colors.RESET}\n")

    def print_help(self):
        """Print help message"""
        help_text = f"""
{Colors.BOLD}COMMANDS:{Colors.RESET}

{Colors.CYAN}Model & Setup:{Colors.RESET}
  load <model.gguf> [tokenizer.json]  Load model (and optionally tokenizer)
  tokenizer <tokenizer.json>          Load tokenizer separately
  compile                             Compile transformer.cu
  status                              Show current status

{Colors.CYAN}Generation:{Colors.RESET}
  chat <message>                      Generate response (or just type message)
  generate <prompt>                   Alias for chat
  run <prompt> [tokens] [temp]        Run with specific settings

{Colors.CYAN}Settings:{Colors.RESET}
  set temperature <0.0-2.0>           Set sampling temperature
  set max_tokens <n>                  Set max tokens to generate
  set top_k <n>                       Set top-k sampling (-1 to disable)
  set top_p <0.0-1.0>                 Set nucleus sampling
  set repetition_penalty <n>          Set repetition penalty
  set context_length <n>              Set context window size
  set seed <n>                        Set random seed (or 'none')
  set device <id>                     Set GPU device ID
  set cpu_layers <0,1,2...>           Set CPU offload layers
  set all_cpu <on/off>                Run entirely on CPU
  set verbose <on/off>                Enable verbose output
  settings                            Show all current settings

{Colors.CYAN}File Operations:{Colors.RESET}
  read <file>                         Read and display file contents
  write <file> [content]              Write to file (or last output if no content)
  append <file> [content]             Append to file
  save <file>                         Save last generated output to file
  list [path]                         List files in directory
  cat <file>                          Alias for read

{Colors.CYAN}Model Inspection:{Colors.RESET}
  info                                Show model information
  tensors                             List all model tensors
  quant                               Show quantization statistics
  benchmark                           Run benchmark test

{Colors.CYAN}Utility:{Colors.RESET}
  history                             Show conversation history
  clear                               Clear conversation history
  log [file]                          Save session log to file
  shell <command>                     Execute shell command
  !<command>                          Shortcut for shell command
  help                                Show this help message
  exit / quit / q                     Exit the GUI

{Colors.DIM}Tip: Just type a message to chat with the model directly.{Colors.RESET}
"""
        print(help_text)

    def compile(self):
        """Compile transformer.cu"""
        src = CONFIG['transformer_src']
        out = CONFIG['transformer_bin']
        flags = CONFIG['compile_flags']
        
        if not os.path.exists(src):
            print(f"{Colors.RED}Error: {src} not found{Colors.RESET}")
            return False
            
        print(f"{Colors.YELLOW}Compiling {src}...{Colors.RESET}")
        cmd = f"nvcc {flags} -o {out} {src}"
        print(f"{Colors.DIM}$ {cmd}{Colors.RESET}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{Colors.GREEN}✓ Compilation successful{Colors.RESET}")
            if result.stderr:
                # Show warnings
                for line in result.stderr.split('\n'):
                    if 'warning' in line.lower():
                        print(f"{Colors.YELLOW}  {line}{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}✗ Compilation failed{Colors.RESET}")
            print(result.stderr)
            return False

    def load_model(self, model_path, tokenizer_path=None):
        """Load model and optionally tokenizer"""
        if not os.path.exists(model_path):
            print(f"{Colors.RED}Error: Model file not found: {model_path}{Colors.RESET}")
            return False
            
        self.model_path = os.path.abspath(model_path)
        print(f"{Colors.GREEN}✓ Model loaded: {os.path.basename(model_path)}{Colors.RESET}")
        
        if tokenizer_path:
            return self.load_tokenizer(tokenizer_path)
        return True
        
    def load_tokenizer(self, tokenizer_path):
        """Load tokenizer"""
        if not os.path.exists(tokenizer_path):
            print(f"{Colors.RED}Error: Tokenizer file not found: {tokenizer_path}{Colors.RESET}")
            return False
            
        self.tokenizer_path = os.path.abspath(tokenizer_path)
        print(f"{Colors.GREEN}✓ Tokenizer loaded: {os.path.basename(tokenizer_path)}{Colors.RESET}")
        self.save_config()
        return True

    def build_command(self, prompt, **kwargs):
        """Build transformer command with all arguments"""
        if not os.path.exists(CONFIG['transformer_bin']):
            print(f"{Colors.RED}Error: Transformer not compiled. Run 'compile' first.{Colors.RESET}")
            return None
            
        if not self.model_path:
            print(f"{Colors.RED}Error: No model loaded. Use 'load <model.gguf>'{Colors.RESET}")
            return None
            
        if not self.tokenizer_path:
            print(f"{Colors.RED}Error: No tokenizer loaded. Use 'tokenizer <tokenizer.json>'{Colors.RESET}")
            return None
        
        # Merge settings with kwargs
        settings = {**self.settings, **kwargs}
        
        cmd = [f'./{CONFIG["transformer_bin"]}', self.model_path, self.tokenizer_path]
        cmd.extend(['-p', prompt])
        cmd.extend(['-n', str(settings.get('max_tokens', 50))])
        cmd.extend(['-t', str(settings.get('temperature', 0.8))])
        
        if settings.get('top_k', -1) != -1:
            cmd.extend(['--top-k', str(settings['top_k'])])
        if settings.get('top_p', 1.0) != 1.0:
            cmd.extend(['--top-p', str(settings['top_p'])])
        if settings.get('repetition_penalty', 1.0) != 1.0:
            cmd.extend(['--repetition-penalty', str(settings['repetition_penalty'])])
        if settings.get('context_length', 1024) != 1024:
            cmd.extend(['--context-length', str(settings['context_length'])])
        if settings.get('seed') is not None:
            cmd.extend(['--seed', str(settings['seed'])])
        if settings.get('cpu_layers'):
            cmd.extend(['--cpu-layers', settings['cpu_layers']])
        if settings.get('all_cpu'):
            cmd.append('--all-cpu')
        if settings.get('device', 0) != 0:
            cmd.extend(['--device', str(settings['device'])])
        if settings.get('verbose'):
            cmd.append('--verbose')
            
        cmd.append('--no-quant-stats')  # Cleaner output for chat
        
        return cmd

    def generate(self, prompt, **kwargs):
        """Generate text from prompt"""
        cmd = self.build_command(prompt, **kwargs)
        if not cmd:
            return None
            
        print(f"\n{Colors.DIM}Running inference...{Colors.RESET}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse output to extract generated text
            output = result.stdout
            
            # Find the generated text section
            generated = ""
            in_generated = False
            for line in output.split('\n'):
                if 'GENERATED TEXT:' in line:
                    in_generated = True
                    continue
                if in_generated:
                    if line.strip() == '=' * 40:
                        continue
                    generated += line + '\n'
                    
            generated = generated.strip()
            
            if not generated:
                # Fallback: look for [AGENT] Output
                for line in output.split('\n'):
                    if '[AGENT] Output:' in line:
                        generated = line.split('[AGENT] Output:')[1].strip()
                        break
            
            if generated:
                self.last_output = generated
                self.conversation_history.append({
                    'prompt': prompt,
                    'response': generated,
                    'timestamp': datetime.now().isoformat()
                })
                return generated
            else:
                print(f"{Colors.YELLOW}No output generated{Colors.RESET}")
                if result.stderr:
                    print(f"{Colors.RED}{result.stderr}{Colors.RESET}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}Error: Generation timed out{Colors.RESET}")
            return None
        except Exception as e:
            print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
            return None

    def run_transformer_command(self, args):
        """Run transformer with arbitrary arguments"""
        if not os.path.exists(CONFIG['transformer_bin']):
            print(f"{Colors.RED}Error: Transformer not compiled{Colors.RESET}")
            return
            
        cmd = [f'./{CONFIG["transformer_bin"]}']
        if self.model_path:
            cmd.append(self.model_path)
        if self.tokenizer_path:
            cmd.append(self.tokenizer_path)
        cmd.extend(args)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"{Colors.YELLOW}{result.stderr}{Colors.RESET}")

    def read_file(self, filepath):
        """Read and display file contents"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            print(f"\n{Colors.DIM}─── {filepath} ───{Colors.RESET}")
            print(content)
            print(f"{Colors.DIM}─── end ───{Colors.RESET}\n")
            return content
        except Exception as e:
            print(f"{Colors.RED}Error reading file: {e}{Colors.RESET}")
            return None

    def write_file(self, filepath, content=None, append=False):
        """Write content to file"""
        if content is None:
            if self.last_output:
                content = self.last_output
            else:
                print(f"{Colors.RED}Error: No content to write{Colors.RESET}")
                return False
                
        try:
            mode = 'a' if append else 'w'
            with open(filepath, mode) as f:
                f.write(content)
                if append and not content.endswith('\n'):
                    f.write('\n')
            action = "Appended to" if append else "Wrote to"
            print(f"{Colors.GREEN}✓ {action} {filepath}{Colors.RESET}")
            return True
        except Exception as e:
            print(f"{Colors.RED}Error writing file: {e}{Colors.RESET}")
            return False

    def list_files(self, path='.'):
        """List files in directory"""
        try:
            entries = sorted(os.listdir(path))
            print(f"\n{Colors.BOLD}Contents of {path}:{Colors.RESET}")
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    print(f"  {Colors.BLUE}{entry}/{Colors.RESET}")
                elif entry.endswith('.gguf'):
                    size = os.path.getsize(full_path) / (1024*1024)
                    print(f"  {Colors.GREEN}{entry}{Colors.RESET} ({size:.1f} MB)")
                elif entry.endswith('.cu'):
                    print(f"  {Colors.CYAN}{entry}{Colors.RESET}")
                else:
                    print(f"  {entry}")
            print()
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")

    def show_settings(self):
        """Display all current settings"""
        print(f"\n{Colors.BOLD}Current Settings:{Colors.RESET}")
        print(f"  temperature:        {self.settings['temperature']}")
        print(f"  max_tokens:         {self.settings['max_tokens']}")
        print(f"  top_k:              {self.settings['top_k']} {'(disabled)' if self.settings['top_k'] == -1 else ''}")
        print(f"  top_p:              {self.settings['top_p']}")
        print(f"  repetition_penalty: {self.settings['repetition_penalty']}")
        print(f"  context_length:     {self.settings['context_length']}")
        print(f"  seed:               {self.settings['seed'] or 'random'}")
        print(f"  device:             {self.settings['device']}")
        print(f"  cpu_layers:         {self.settings['cpu_layers'] or 'none'}")
        print(f"  all_cpu:            {'on' if self.settings['all_cpu'] else 'off'}")
        print(f"  verbose:            {'on' if self.settings['verbose'] else 'off'}")
        print()

    def set_setting(self, key, value):
        """Set a configuration value"""
        try:
            if key == 'temperature':
                self.settings['temperature'] = float(value)
            elif key == 'max_tokens':
                self.settings['max_tokens'] = int(value)
            elif key == 'top_k':
                self.settings['top_k'] = int(value)
            elif key == 'top_p':
                self.settings['top_p'] = float(value)
            elif key == 'repetition_penalty':
                self.settings['repetition_penalty'] = float(value)
            elif key == 'context_length':
                self.settings['context_length'] = int(value)
            elif key == 'seed':
                self.settings['seed'] = None if value.lower() == 'none' else int(value)
            elif key == 'device':
                self.settings['device'] = int(value)
            elif key == 'cpu_layers':
                self.settings['cpu_layers'] = value if value.lower() != 'none' else None
            elif key == 'all_cpu':
                self.settings['all_cpu'] = value.lower() in ('on', 'true', '1', 'yes')
            elif key == 'verbose':
                self.settings['verbose'] = value.lower() in ('on', 'true', '1', 'yes')
            else:
                print(f"{Colors.RED}Unknown setting: {key}{Colors.RESET}")
                return
                
            print(f"{Colors.GREEN}✓ Set {key} = {value}{Colors.RESET}")
            self.save_config()
        except ValueError as e:
            print(f"{Colors.RED}Invalid value: {e}{Colors.RESET}")

    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print(f"{Colors.YELLOW}No conversation history{Colors.RESET}")
            return
            
        print(f"\n{Colors.BOLD}Conversation History:{Colors.RESET}")
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{Colors.CYAN}[{i}] {entry['timestamp']}{Colors.RESET}")
            print(f"{Colors.BOLD}You:{Colors.RESET} {entry['prompt'][:100]}{'...' if len(entry['prompt']) > 100 else ''}")
            print(f"{Colors.GREEN}Bot:{Colors.RESET} {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}")
        print()

    def save_log(self, filepath=None):
        """Save session log"""
        filepath = filepath or CONFIG['log_file']
        try:
            with open(filepath, 'w') as f:
                f.write(f"Transformer GUI Session Log\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Tokenizer: {self.tokenizer_path}\n")
                f.write("="*60 + "\n\n")
                
                for entry in self.conversation_history:
                    f.write(f"[{entry['timestamp']}]\n")
                    f.write(f"Prompt: {entry['prompt']}\n")
                    f.write(f"Response: {entry['response']}\n")
                    f.write("-"*40 + "\n\n")
                    
            print(f"{Colors.GREEN}✓ Log saved to {filepath}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error saving log: {e}{Colors.RESET}")

    def run_shell(self, command):
        """Execute shell command"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"{Colors.YELLOW}{result.stderr}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")

    def process_command(self, user_input):
        """Process user command"""
        user_input = user_input.strip()
        if not user_input:
            return True
            
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Exit commands
        if cmd in ('exit', 'quit', 'q'):
            return False
            
        # Help
        elif cmd == 'help':
            self.print_help()
            
        # Status
        elif cmd == 'status':
            self.print_status()
            
        # Compile
        elif cmd == 'compile':
            self.compile()
            
        # Load model
        elif cmd == 'load':
            if not args:
                print(f"{Colors.RED}Usage: load <model.gguf> [tokenizer.json]{Colors.RESET}")
            else:
                load_args = args.split()
                model = load_args[0]
                tok = load_args[1] if len(load_args) > 1 else None
                self.load_model(model, tok)
                
        # Load tokenizer
        elif cmd == 'tokenizer':
            if not args:
                print(f"{Colors.RED}Usage: tokenizer <tokenizer.json>{Colors.RESET}")
            else:
                self.load_tokenizer(args)
                
        # Generate / Chat
        elif cmd in ('chat', 'generate', 'run'):
            if not args:
                print(f"{Colors.RED}Usage: {cmd} <prompt>{Colors.RESET}")
            else:
                # Parse optional tokens and temp for 'run' command
                if cmd == 'run':
                    run_parts = args.split()
                    if len(run_parts) >= 3:
                        prompt = ' '.join(run_parts[:-2])
                        tokens = int(run_parts[-2])
                        temp = float(run_parts[-1])
                        result = self.generate(prompt, max_tokens=tokens, temperature=temp)
                    elif len(run_parts) >= 2 and run_parts[-1].replace('.','').isdigit():
                        prompt = ' '.join(run_parts[:-1])
                        tokens = int(run_parts[-1])
                        result = self.generate(prompt, max_tokens=tokens)
                    else:
                        result = self.generate(args)
                else:
                    result = self.generate(args)
                    
                if result:
                    print(f"\n{Colors.GREEN}{Colors.BOLD}Response:{Colors.RESET}")
                    print(f"{result}\n")
                    
        # Settings
        elif cmd == 'set':
            set_parts = args.split(maxsplit=1)
            if len(set_parts) < 2:
                print(f"{Colors.RED}Usage: set <setting> <value>{Colors.RESET}")
            else:
                self.set_setting(set_parts[0], set_parts[1])
                
        elif cmd == 'settings':
            self.show_settings()
            
        # File operations
        elif cmd in ('read', 'cat'):
            if not args:
                print(f"{Colors.RED}Usage: {cmd} <file>{Colors.RESET}")
            else:
                self.read_file(args)
                
        elif cmd == 'write':
            write_parts = args.split(maxsplit=1)
            if not write_parts:
                print(f"{Colors.RED}Usage: write <file> [content]{Colors.RESET}")
            else:
                filepath = write_parts[0]
                content = write_parts[1] if len(write_parts) > 1 else None
                self.write_file(filepath, content)
                
        elif cmd == 'append':
            write_parts = args.split(maxsplit=1)
            if not write_parts:
                print(f"{Colors.RED}Usage: append <file> [content]{Colors.RESET}")
            else:
                filepath = write_parts[0]
                content = write_parts[1] if len(write_parts) > 1 else None
                self.write_file(filepath, content, append=True)
                
        elif cmd == 'save':
            if not args:
                print(f"{Colors.RED}Usage: save <file>{Colors.RESET}")
            else:
                self.write_file(args)
                
        elif cmd in ('list', 'ls'):
            self.list_files(args if args else '.')
            
        # Model inspection
        elif cmd == 'info':
            self.run_transformer_command(['-i'])
            
        elif cmd == 'tensors':
            self.run_transformer_command(['--list-tensors'])
            
        elif cmd == 'quant':
            self.run_transformer_command(['--show-quant-stats'])
            
        elif cmd == 'benchmark':
            self.run_transformer_command(['--benchmark'])
            
        # History
        elif cmd == 'history':
            self.show_history()
            
        elif cmd == 'clear':
            self.conversation_history = []
            print(f"{Colors.GREEN}✓ History cleared{Colors.RESET}")
            
        elif cmd == 'log':
            self.save_log(args if args else None)
            
        # Shell
        elif cmd == 'shell' or user_input.startswith('!'):
            shell_cmd = args if cmd == 'shell' else user_input[1:]
            self.run_shell(shell_cmd)
            
        # Default: treat as chat message
        else:
            result = self.generate(user_input)
            if result:
                print(f"\n{Colors.GREEN}{Colors.BOLD}Response:{Colors.RESET}")
                print(f"{result}\n")
                
        return True

    def run(self):
        """Main loop"""
        self.print_header()
        self.print_status()
        
        print(f"{Colors.DIM}Type 'help' for commands, or just type a message to chat.{Colors.RESET}\n")
        
        try:
            while True:
                try:
                    prompt = f"{Colors.BOLD}>{Colors.RESET} "
                    user_input = input(prompt)
                    
                    if not self.process_command(user_input):
                        break
                        
                except KeyboardInterrupt:
                    print(f"\n{Colors.YELLOW}Use 'exit' to quit{Colors.RESET}")
                    continue
                    
        except EOFError:
            pass
        finally:
            self.save_history()
            self.save_config()
            print(f"\n{Colors.CYAN}Goodbye!{Colors.RESET}\n")


def main():
    # Check if transformer source exists
    if not os.path.exists(CONFIG['transformer_src']):
        print(f"{Colors.RED}Error: {CONFIG['transformer_src']} not found in current directory{Colors.RESET}")
        sys.exit(1)
        
    gui = TransformerGUI()
    gui.run()


if __name__ == '__main__':
    main()
