# To look up the sense of words based on the sentence contexts
# To manually check whether the automatically identified sense is correct or not
# If not, manually identify the sense of each word

from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

noun_responses = []
verb_responses = []
noun_senses_lists = []
verb_senses_lists = []

# Just to make it easier to read
class color:
   PURPLE = ''
   CYAN = ''
   DARKCYAN = ''
   BLUE = ''
   GREEN = ''
   YELLOW = ''
   RED = ''
   BOLD = ''
   UNDERLINE = ''
   END = ''

contexts = []

#Context sentences taken from Final_by-items_Lexical_variables_verbs_Final_All_Fillers.xlsx
verbs = ["blindfolded", "enraged"]
words = ["wife", "judge"]
sentences = ["After ten years of marriage, Ted finally bought his dream house  He wanted to break the news as a surprise he blindfolded his wife", "The defendant appeared to be drunk during court  Even after several warnings, he refused to remain silent during the proceeding he enraged his judge"]

for count, word in enumerate(words):
    this_noun_synset = lesk(sentences[count], word, 'n')
    this_verb_synset = lesk(sentences[count], verbs[count], 'v')
    print("\n\n\n+++++++ " + color.BOLD + sentences[count] + color.END + " +++++++")
    print("\nNoun is " + color.BOLD + word + color.END + ", as in " + this_noun_synset.definition())
    try:
         print("\nExample: " + this_noun_synset.examples()[0])
    except:
        print("no examples of noun:(")
    print("\nVerb is " + color.BOLD + verbs[count] + color.END + ", as in " + this_verb_synset.definition())
    try:
         print("\nExample: " + this_verb_synset.examples()[0])
    except:
        print("no examples of verb :(")
    
    # prepend chosen sense, so that 0 is the original prediction
    noun_senses_lists.append([this_noun_synset] + wn.synsets(word))
    verb_senses_lists.append([this_verb_synset] + wn.synsets(verbs[count]))
    
    # print("This is where the words appears: " + color.GREEN + sentences[count] + color.END)
    
    verb_option_counter = 0
    print("\n--- Senses for verb: " + verbs[count] + " ---")
    print("\n0 (chosen): " + this_verb_synset.definition() + "\n")
    for sense in wn.synsets(verbs[count]):
        verb_option_counter = verb_option_counter + 1
        print(str(verb_option_counter) + color.PURPLE + " (" + str(sense) + "): " + sense.definition() + color.END)
    verb_responses.append(input("\nWhich number best represents the verb?"))

    noun_option_counter = 0
    print("\n--- Senses for noun: " + word + " ---")
    print("\n0 (chosen): " + this_noun_synset.definition() + "\n")
    for sense in wn.synsets(word):
        noun_option_counter = noun_option_counter + 1
        print(str(noun_option_counter) + color.PURPLE + " (" + str(sense) + "): " + sense.definition() + color.END)
    noun_responses.append(input("\nWhich number best represents the noun?"))

    final_noun_list = []
    final_verb_list = []
    for count, response in enumerate(noun_responses):
        try:
            final_noun_list.append(noun_senses_lists[count][int(response)])
        except:
            final_noun_list.append("NaN")
    for count, response in enumerate(verb_responses):
        try:
            final_verb_list.append(verb_senses_lists[count][int(response)])
        except:
            final_verb_list.append("NaN")
            
    print("List so far: " + color.CYAN + str(final_noun_list) + str(final_verb_list) + color.END)
    with open("output_of_sense_choices.txt", "w") as f:
        f.write("Nouns: " + str(final_noun_list))
        f.write("\nVerbs: " + str(final_verb_list))
        f.write("\nBackup nouns: " + str(noun_responses))
        f.write("\nBackup verbs: " + str(verb_responses))
        



