import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string
import numpy as np
from datetime import datetime

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading necessary NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text by removing special characters, numbers, and converting to lowercase.
    
    Args:
        text (str): The input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords_tfidf(text, num_keywords=15):
    """
    Extract keywords using TF-IDF.
    
    Args:
        text (str): The input text
        num_keywords (int): Number of keywords to extract
        
    Returns:
        list: List of (keyword, score) tuples
    """
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords
    custom_stopwords = {'also', 'would', 'could', 'may', 'might', 'must', 'need',
                        'shall', 'should', 'will', 'would', 'one', 'two', 'three',
                        'first', 'second', 'third', 'new', 'old', 'time', 'year',
                        'day', 'today', 'tomorrow', 'month', 'date', 'copyright',
                        'rights', 'reserved', 'privacy', 'policy', 'terms', 'use',
                        'contact', 'us', 'email', 'phone', 'address', 'website',
                        'click', 'link', 'page', 'site', 'web', 'visit', 'view',
                        'read', 'see', 'go', 'get', 'find', 'look', 'search',
                        'know', 'make', 'take', 'give', 'use', 'work', 'come'}
    
    stop_words.update(custom_stopwords)
    
    # Tokenize and filter
    tokens = word_tokenize(processed_text)
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Join back into a single string for TF-IDF
    processed_text = ' '.join(lemmatized_tokens)
    
    # If text is too short after processing, return empty list
    if len(processed_text.split()) < 5:
        return []
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=num_keywords*2)
    
    try:
        # Fit and transform the text
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        
        # Get feature names and TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Create a list of (keyword, score) tuples and sort by score
        keyword_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return keyword_scores[:num_keywords]
    
    except ValueError:
        # If vectorizer fails (e.g., empty text after preprocessing)
        return []

def extract_phrases(text, num_phrases=5, min_count=2):
    """
    Extract common phrases from text.
    
    Args:
        text (str): The input text
        num_phrases (int): Number of phrases to extract
        min_count (int): Minimum count for a phrase to be considered
        
    Returns:
        list: List of (phrase, count) tuples
    """
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(processed_text)
    
    # Extract 2-5 word phrases
    phrases = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        
        # Skip short sentences
        if len(words) < 3:
            continue
        
        # Extract phrases of different lengths
        for phrase_len in range(2, 6):
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i+phrase_len])
                # Only include phrases without stopwords at the beginning or end
                if len(phrase) > 5 and not any(phrase.startswith(sw + ' ') or phrase.endswith(' ' + sw) 
                                              for sw in stopwords.words('english')):
                    phrases.append(phrase)
    
    # Count phrases
    phrase_counts = Counter(phrases)
    
    # Filter by minimum count and sort
    common_phrases = [(phrase, count) for phrase, count in phrase_counts.items() if count >= min_count]
    common_phrases.sort(key=lambda x: x[1], reverse=True)
    
    return common_phrases[:num_phrases]

def extract_named_entities(text, num_entities=5):
    """
    Extract potential named entities using simple heuristics.
    
    Args:
        text (str): The input text
        num_entities (int): Number of entities to extract
        
    Returns:
        list: List of potential named entities
    """
    # Look for capitalized words that aren't at the beginning of sentences
    sentences = sent_tokenize(text)
    potential_entities = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        
        # Skip the first word of each sentence
        for i in range(1, len(words)):
            word = words[i]
            # Check if word is capitalized and not a stopword
            if (word[0].isupper() and word.lower() not in stopwords.words('english') 
                and len(word) > 1 and word not in string.punctuation):
                potential_entities.append(word)
    
    # Count entities
    entity_counts = Counter(potential_entities)
    
    # Get most common entities
    common_entities = entity_counts.most_common(num_entities)
    
    return common_entities

def process_text_files(directory="scraped_data", output_file=None):
    """
    Process all text files in a directory and extract keywords.
    
    Args:
        directory (str): Directory containing text files
        output_file (str): Output file path for the spreadsheet
        
    Returns:
        pandas.DataFrame: DataFrame containing the results
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return None
    
    # Get all text files in the directory
    text_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    if not text_files:
        print(f"No text files found in '{directory}'.")
        return None
    
    print(f"Found {len(text_files)} text files in '{directory}'.")
    
    # Create a list to store results
    results = []
    
    # Process each file
    for i, file_name in enumerate(text_files):
        file_path = os.path.join(directory, file_name)
        print(f"Processing file {i+1}/{len(text_files)}: {file_name}")
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract source URL if available
            source_url = ""
            url_match = re.search(r'Source URL: (https?://\S+)', text)
            if url_match:
                source_url = url_match.group(1)
                # Remove the metadata header from the text
                text = re.sub(r'Source URL: .*\n.*\n=+\n\n', '', text)
            
            # Extract keywords using TF-IDF
            keywords = extract_keywords_tfidf(text, num_keywords=15)
            
            # Extract common phrases
            phrases = extract_phrases(text, num_phrases=5)
            
            # Extract potential named entities
            entities = extract_named_entities(text, num_entities=5)
            
            # Calculate text statistics
            word_count = len(re.findall(r'\b\w+\b', text))
            sentence_count = len(sent_tokenize(text))
            
            # Format keywords as string
            keywords_str = ', '.join([f"{kw} ({score:.3f})" for kw, score in keywords])
            
            # Format phrases as string
            phrases_str = ', '.join([f"{phrase} ({count})" for phrase, count in phrases])
            
            # Format entities as string
            entities_str = ', '.join([f"{entity} ({count})" for entity, count in entities])
            
            # Add to results
            results.append({
                'File Name': file_name,
                'Source URL': source_url,
                'Word Count': word_count,
                'Sentence Count': sentence_count,
                'Keywords (TF-IDF)': keywords_str,
                'Common Phrases': phrases_str,
                'Potential Named Entities': entities_str
            })
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(f"Error processing file {file_name}: {error_msg}")
            results.append({
                'File Name': file_name,
                'Source URL': '',
                'Word Count': 0,
                'Sentence Count': 0,
                'Keywords (TF-IDF)': "ERROR - See console output",
                'Common Phrases': '',
                'Potential Named Entities': ''
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to file if output_file is provided
    if output_file:
        try:
            # Determine file extension
            _, ext = os.path.splitext(output_file)
            
            if ext.lower() == '.xlsx':
                # Use a try-except block to handle potential Excel writing errors
                try:
                    df.to_excel(output_file, index=False)
                    print(f"Results saved to Excel file: {output_file}")
                except Exception as e:
                    print(f"Error saving to Excel: {str(e)}")
                    # Fall back to CSV if Excel fails
                    csv_output = output_file.replace('.xlsx', '.csv')
                    df.to_csv(csv_output, index=False)
                    print(f"Results saved to CSV file instead: {csv_output}")
            else:
                # Default to CSV
                df.to_csv(output_file, index=False)
                print(f"Results saved to CSV file: {output_file}")
        except Exception as e:
            print(f"Error saving output file: {str(e)}")
            # Save to a default CSV as fallback
            df.to_csv("keyword_extraction_results.csv", index=False)
            print("Results saved to fallback file: keyword_extraction_results.csv")
    
    return df

def main():
    # Make sure all required NLTK data is downloaded
    print("Ensuring all required NLTK resources are available...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    print("\n" + "=" * 60)
    print("Text File Keyword Extractor".center(60))
    print("=" * 60 + "\n")
    
    # Get input directory
    directory = input("Enter the directory containing text files (default: scraped_data): ").strip()
    if not directory:
        directory = "scraped_data"
    
    # Get output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"keywords_{timestamp}.xlsx"
    
    output_file = input(f"Enter the output file name (default: {default_output}): ").strip()
    if not output_file:
        output_file = default_output
    
    # Add extension if not provided
    if not output_file.endswith(('.csv', '.xlsx')):
        output_file += '.xlsx'
    
    print(f"\nProcessing text files from '{directory}'...")
    print(f"Results will be saved to '{output_file}'")
    
    # Process files
    df = process_text_files(directory, output_file)
    
    if df is not None:
        print("\nKeyword extraction completed successfully!")
        print(f"Processed {len(df)} files.")
        
        # Display a sample of the results
        print("\nSample of extracted keywords:")
        for i, row in df.head(3).iterrows():
            print(f"\nFile: {row['File Name']}")
            print(f"Keywords: {row['Keywords (TF-IDF)'][:100]}...")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

# Test the script with a sample text
sample_text = """
Database Management System (DBMS) is a software for storing and retrieving users' data while considering appropriate security measures. It consists of a group of programs which manipulate the database. The DBMS accepts the request for data from an application and instructs the operating system to provide the specific data. In large systems, a DBMS helps users and other third-party software to store and retrieve data.

DBMS allows users to create their own databases as per their requirement. The term "DBMS" includes the user of the database and other application programs. It provides an interface between the data and the software application.
"""

print("\nTesting with sample text:")
keywords = extract_keywords_tfidf(sample_text)
print("Sample Keywords:")
for kw, score in keywords:
    print(f"- {kw}: {score:.3f}")