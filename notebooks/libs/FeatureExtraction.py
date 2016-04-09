# coding: utf-8
import scipy
from lxml import etree


# returns time represented as seconds from hh:mm format
def parse_time(time_string):
    hh = time_string.split(":")[0];
    mm = time_string.split(":")[1];
    hh = int(hh)*3600;
    mm = int(mm)*60;
    return hh+mm;


#returns the duration of given conversation, in seconds
def chat_duration(conversation): 
    times=conversation.xpath('./message/time')
    start=parse_time(extract_text_between_tags(times[0]))
    end=parse_time(extract_text_between_tags(times[len(times)-1]))
    return end-start;


#converts time in seconds to string in hh:mm format
def seconds_to_string(time):
    str=""
    hh = time/3600
    print hh
    mm = time/float(60)%60
    if(hh<10):
        str+="0{}".format(hh)
    else:
        str+="{}".format(hh)
    str+=":"
    if(mm<10):
        str+="0{}".format(int(mm))
    else:
        str+="{}".format(int(mm))
    return str


def extract_author_text_dictionary_from_message_nodes(message_nodes):
    author_text_dictionary = {}
    for message_node in message_nodes:
        text = message_node.xpath('./text')[0].text      
        author_id = message_node.xpath('./author')[0].text
   
        author_text_dictionary.setdefault(author_id, []).append(text)
        
    return author_text_dictionary


def extract_authors_from_parsed_text(parsed_text):
    author_nodes = parsed_text.xpath('/conversations/conversation/message/author')
    unique_authors = set()
    for author_node in author_nodes:
        unique_authors.add(extract_text_between_tags(author_node))
    return unique_authors


def extract_author_nodes_as_list_from_parsed_text(parsed_text):
    return parsed_text.xpath('/conversations/conversation/message/author')


def extract_message_nodes_as_list_from_parsed_text(parsed_text):
    return parsed_text.xpath('/conversations/conversation/message')


def first_message_node_in_conversation(conversation):
    return conversation.xpath(".//message[1]")[0]


def last_message_node_in_conversation(conversation):
    return conversation.xpath(".//message[last()]")[0]


def first_time_node_in_conversation(conversation):
    return first_message_node_in_conversation(conversation).xpath(".//time[1]")[0]


def last_time_node_in_conversation(conversation):
    return last_message_node_in_conversation(conversation).xpath(".//time[1]")[0]


def all_time_nodes_in_conversation(conversation):
    return conversation.xpath("./message/time")


def all_time_nodes_of_author_in_conversation(conversation,author_id):
    xstring="./message[author='{author}']/time".format(author=author_id)
    tekstovi=[t.text for t in conversation.xpath(xstring)]
    return tekstovi


def n_of_people_in_chat(conversation,author):
    return len(ids_of_authors_in_chat(conversation))


def ids_of_authors_in_chat(conversation):
    return set([s.text for s in conversation.xpath('./message/author')])


#returns dict keys are ids of authors involved and the values are average time in seconds between two proceeding lines
def avg_times_between_message_lines_for_all_authors_in_seconds(conversation):
    author_times={}
    authors=ids_of_authors_in_chat(conversation)
    for author in authors:
        author_times[author]=avg_time_between_message_lines_in_seconds_for_author(conversation,author)
    return author_times


#returns average time in seconds between two proceeding lines in given conversation for given author_id
def avg_time_between_message_lines_in_seconds_for_author_in_conversation(conversation,author_id):
    times=all_time_nodes_of_author_in_conversation(conversation,author_id)
    if len(times)==0:
        return None
    return (parse_time(times[-1].text)-parse_time(times[0].text))/(len(times)-1)


def conversation_length_in_seconds(conversation,autor_id=None):
    times=all_time_nodes_in_conversation(conversation)
    return (parse_time(last_time_node_in_conversation(conversation).text)-parse_time(first_time_node_in_conversation(conversation).text))


def percentage_of_lines_in_conversation(conversation,author_id):
    return len(all_time_nodes_of_author_in_conversation(conversation,author_id))*1.0/len(all_time_nodes_in_conversation(conversation))
 

def message_texts_in_conversation(conversation):
    return [t.text for t in conversation.xpath('./message/text')]


def message_texts_of_author_in_conversation(conversation,author_id):
    xstring="./message[author='{author}']/text".format(author=author_id)
    text_nodes=[t.text for t in conversation.xpath(xstring)]
    return text_nodes


def percentage_of_characters_in_conversation(conversation,author_id):
    authors_chars=''.join(messages_of_author_in_conversation(conversation,author_id))
    all_chats=''.join(message_texts_in_conversation(conversation))
    return len(authors_chars)*1.0/len(all_chats)


def is_starting_conversation(conversation,author_id):
    k=first_message_node_in_conversation(conversation).xpath("./author")
    if k[0].text == author_id:
        return True
    return False


def all_conversation_nodes_of_author(tree,author_id):
    xstring="/conversations/conversation/message[author='{author}']/..".format(author=author_id)
    return tree.xpath(xstring)


#function makes average of another function result over all conversations of given author
#argument funct is expexted to be a function(conversation,author_id) type, for example 
# avg_time_between_message_lines_in_seconds_for_author_in_conversation
def average_trough_all_conversations(tree,author_id,funct):
    conversations=all_conversation_nodes_of_author(tree,author_id)
    results=[funct(c,author_id) for c in conversations]
    return sum(results)/len(results)

def extract_conversation_nodes_as_list_from_parsed_text(xml):
    return parsed_text.xpath('/conversations/conversation')


def extract_author_conversation_node_dictionary_from_XML(xml):
    conversation_nodes = extract_conversation_nodes_as_list_from_parsed_text(xml)
    author_text_dictionary = {}
    for conversation_node in conversation_nodes:
        author_id = conversation_node.xpath('./message/author')[0].text       
        author_text_dictionary.setdefault(author_id, []).append(conversation_node)
        
    return author_text_dictionary


def sexual_predator_ids(filePath):
    with open(filePath) as f:
        return f.read().splitlines()

