def split_file(input_file, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    try:
        # Read all lines
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Calculate split points
        train_end = int(total_lines * train_ratio)
        dev_end = train_end + int(total_lines * dev_ratio)
        
        # Split data
        train_data = lines[:train_end]
        dev_data = lines[train_end:dev_end]
        test_data = lines[dev_end:]
        
        # Write splits to files
        base_name = input_file.rsplit('.', 1)[0]
        
        with open(f"{base_name}_train.txt", 'w', encoding='utf-8') as f:
            f.writelines(train_data)
            
        with open(f"{base_name}_dev.txt", 'w', encoding='utf-8') as f:
            f.writelines(dev_data)
            
        with open(f"{base_name}_test.txt", 'w', encoding='utf-8') as f:
            f.writelines(test_data)
            
        print(f"Split complete: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)} lines")
        
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    split_file("./raw.kb")
