# coding: utf-8

from lxml import etree
import FeatureExtraction as FE
import numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


sexual_predator_ids_file = '../../dataset/training/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'

def create_csv(input_file_path, output_file_name, batch_size):
    tree = etree.parse(input_file_path)
    author_conversation_node_dictionary = FE.extract_author_conversation_node_dictionary_from_XML(tree)
    del tree
    
    output_file_csv = open(output_file_name, 'w+')
    output_string_list = ['autor', 'number of conversation', 'percent of conversations started by the author',
                         'difference between two preceding lines in seconds', 'number of messages sent',
                         'average percent of lines in conversation', 'average percent of characters in conversation',
                         'number of characters sent by the author', 'mean time of messages sent',
                         'number of unique contacted authors', 'avg number of unique authors interacted with per conversation',
                         'total unique authors and unique per chat difference',
                         'conversation num and total unique authors difference',
                         'average question marks per conversations', 'total question marks', 'total author question marks',
                         'avg author question marks', 'author and conversation quetsion mark differnece',
                         #'total negative in author conv', 'total neutral in author conv', 'total positive in author conv',
                         #'total compound in author conv', 'author total negative in author conv',
                         #'author total neutral in author conv', 'author total positive in author conv',
                         #' authortotal compound in author conv', 'conversation and author negative differnece',
                         #'conversation and author neutral differnece', 'conversation and author positive differnece',
                         #' conversation and author compound differnece',
                         'is sexual predator']
    output_string = ','.join(output_string_list) + "\n"
    
    sexual_predator_ids_list = FE.sexual_predator_ids(sexual_predator_ids_file)
    
    i = 0
    for index, author in enumerate(sorted(author_conversation_node_dictionary)):
        
        if index % 100 == 0:
            print index, len(author_conversation_node_dictionary)
        
        conversation_nodes = author_conversation_node_dictionary[author]
        conversation_nodes_length = len(conversation_nodes)
        
        #conversation_text_sentiment_total = FE.calculate_conversation_sentiment_total(
        #    author, conversation_nodes)
        
        #author_conversation_text_sentiment_total = FE.calculate_author_conversation_sentiment_total(author, conversation_nodes)
        
        total_unique_authors = FE.number_of_unique_authors_interacted_with(author, conversation_nodes)
        total_author_question_marks = FE.total_authors_question_marks_per_conversation(author, conversation_nodes)
        
        output_list = [author,
                       len(conversation_nodes),
                       FE.average_trough_all_conversations(author, conversation_nodes, FE.is_starting_conversation),
                       FE.average_trough_all_conversations(author, conversation_nodes, 
                                    FE.avg_time_between_message_lines_in_seconds_for_author_in_conversation),
                       FE.number_of_messages_sent_by_the_author(author, conversation_nodes),
                       FE.average_trough_all_conversations(author, conversation_nodes,
                                                           FE.percentage_of_lines_in_conversation),
                       FE.average_trough_all_conversations(author, conversation_nodes,
                                                           FE.percentage_of_characters_in_conversation),
                       FE.number_of_characters_sent_by_the_author(author, conversation_nodes),
                       FE.mean_time_of_messages_sent(author, conversation_nodes),
                       total_unique_authors,
                       total_unique_authors/conversation_nodes_length,
                       FE.difference_unique_authors_per_chat_and_total_unique(
                           total_unique_authors, total_unique_authors/conversation_nodes_length),
                       FE.difference_unique_authors_and_conversations(
                           total_unique_authors, conversation_nodes_length
                        ),
                       FE.avg_question_marks_per_conversation(author, conversation_nodes),
                       FE.total_question_marks_per_conversation(author, conversation_nodes),
                       total_author_question_marks,
                       total_author_question_marks/conversation_nodes_length,
                       abs(total_author_question_marks - FE.total_question_marks_per_conversation(author, conversation_nodes)),
                       #conversation_text_sentiment_total['neg'],
                       #conversation_text_sentiment_total['neu'],
                       #conversation_text_sentiment_total['pos'],
                       #conversation_text_sentiment_total['compound'],
                       #author_conversation_text_sentiment_total['neg'],
                       #author_conversation_text_sentiment_total['neu'],
                       #author_conversation_text_sentiment_total['pos'],
                       #author_conversation_text_sentiment_total['compound'],
                       #abs(conversation_text_sentiment_total['neg']-author_conversation_text_sentiment_total['neg']),
                       #abs(conversation_text_sentiment_total['neu']-author_conversation_text_sentiment_total['neu']),
                       #abs(conversation_text_sentiment_total['pos']-author_conversation_text_sentiment_total['pos']),
                       #abs(conversation_text_sentiment_total['compound']-author_conversation_text_sentiment_total['compound']),
                       '1' if author in sexual_predator_ids_list else '0'
                      ]
        output_string += ','.join(map(str, output_list)) + '\n'
        if i == batch_size:
            output_file_csv.write(output_string)
            output_string = ''
            i = -1
            
        i += 1
        
    output_file_csv.write(output_string)    
    del output_string
    del author_conversation_node_dictionary
    output_file_csv.close()


file_path='../../dataset/training/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
create_csv(file_path, '../../csv/chat_based_features_training.csv', 1)
