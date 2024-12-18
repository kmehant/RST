import asyncio
import re
import uuid
import logging
import types
from .._grammar import grammar
from .._utils import escape_template_block, strip_markers
import guidance
import re
import json


log = logging.getLogger(__name__)
cnt = 0
end_indent_flag = 0
missed_required_keys = []

def cnt_spaces_right(text):
    return len(text) - len(text.rstrip(' '))

def cnt_spaces_left(text):
    return len(text) - len(text.lstrip(' '))

def return_required_keys(contents, module):
    dic = contents[module]
    required_keys = []
    key_schema = {}
    for key in dic["doc"]["options"]:
        if "required" in dic["doc"]["options"][key].keys():
            if dic["doc"]["options"][key]["required"]==True:
                required_keys.append(key)
                
    for key in required_keys:
        key_schema[key] = []
        if "suboptions" in dic["doc"]["options"][key]:
            for key_1 in dic["doc"]["options"][key]["suboptions"]:
                if "required" in dic["doc"]["options"][key]["suboptions"][key_1] and dic["doc"]["options"][key]["suboptions"][key_1]["required"]==True:
                    key_schema[key].append(key_1)
                
    return(key_schema)

def return_template_elements():
        
        with open('/dccstor/rhassistant/mehant/paper/guidance/guidance-guidanceV2/data_with_ft.jsonl', 'r') as json_file:
          json_list = list(json_file)

        contents = {}
        for json_str in json_list:
          result = json.loads(json_str)
          if result['ansible_module'] in result:
            contents[result['ansible_module']] = result[result['ansible_module']]

        required_keys = return_required_keys(contents, guidance.task_details["module_name"])

        all_keys = list(contents[guidance.task_details["module_name"]]["doc"]["options"].keys())
        optional_keys = []
        for i in all_keys:
          if i not in required_keys:
            optional_keys.append(i)
        return required_keys, optional_keys

def build_template(required_keys):
        # template for ansible is built as all the level 1 keys in select and select in geneach with max_iteration = no. of keys there is gen after every select for value generation 
        max_iter = len(required_keys)
        guidance_template = """{{#geneach "items" min_iterations=0 max_iterations=""" + str(max_iter) +  """ unique=unique}}{{#select "key" id=id}}"""
        for key in required_keys[:-1]:
          guidance_template += key + "{{or}}"

        guidance_template += required_keys[-1] + "{{/select}}:" + """{{gen "value" dic=options_available}}{{/geneach}}"""
        return(guidance_template)


async def gen(name=None, stop=None, stop_regex=None, save_stop_text=False, max_tokens=500, n=1, stream=None,
              temperature=0.0, top_p=1.0, logprobs=None, pattern=None, hidden=False, list_append=False,
              save_prompt=False, token_healing=None, _parser_context=None, dic=None, func_call=None):
    ''' Use the LLM to generate a completion.

    Parameters
    ----------
    name : str or None
        The name of a variable to store the generated value in. If none the value is just returned.
    stop : str
        The stop string to use for stopping generation. If not provided, the next node's text will be used if
        that text matches a closing quote, XML tag, or role end. Note that the stop string is not included in
        the generated value.
    stop_regex : str
        A regular expression to use for stopping generation. If not provided, the stop string will be used.
    save_stop_text : str or bool
        If set to a string, the exact stop text used will be saved in a variable with the given name. If set to
        True, the stop text will be saved in a variable named `name+"_stop_text"`. If set to False,
        the stop text will not be saved.
    max_tokens : int
        The maximum number of tokens to generate in this completion.
    n : int
        The number of completions to generate. If you generate more than one completion, the variable will be
        set to a list of generated values. Only the first completion will be used for future context for the LLM,
        but you may often want to use hidden=True when using n > 1.
    temperature : float
        The temperature to use for generation. A higher temperature will result in more random completions. Note
        that caching is always on for temperature=0, and is seed-based for other temperatures.
    top_p : float
        The top_p value to use for generation. A higher top_p will result in more random completions.
    logprobs : int or None
        If set to an integer, the LLM will return that number of top log probabilities for the generated tokens
        which will be stored in a variable named `name+"_logprobs"`. If set to None, the log
        probabilities will not be returned.
    pattern : str or None
        A regular expression pattern guide to use for generation. If set the LLM will be forced (through guided
        decoding) to only generate completions that match the regular expression.
    hidden : bool
        Whether to hide the generated value from future LLM context. This is useful for generating completions
        that you just want to save in a variable and not use for future context.
    list_append : bool
        Whether to append the generated value to a list stored in the variable. If set to True, the variable
        must be a list, and the generated value will be appended to the list.
    save_prompt : str or bool
        If set to a string, the exact prompt given to the LLM will be saved in a variable with the given name.
    token_healing : bool or None
        If set to a bool this overrides the token_healing setting for the LLM.
    '''
    prefix = ""
    suffix = ""
    global missed_required_keys
    global end_indent_flag

    # get the parser context variables we will need
    parser = _parser_context['parser']
    next_node = _parser_context["next_node"]
    next_next_node = _parser_context["next_next_node"]
    prev_node = _parser_context["prev_node"]
    parser_prefix = _parser_context["parser_prefix"]
    partial_output = _parser_context["partial_output"]
    module_name_spaces = None
    pos = len(parser.prefix) # save the current position in the prefix
    
    if guidance.task_details and guidance.task_details["module_name"]:
        lines = parser_prefix.split("\n")
        for line in lines:
            if(guidance.task_details["module_name"] in line.split(":")[0]):
                module_name_spaces = cnt_spaces_left(line)
                break
    if list_append:
        assert name is not None, "You must provide a variable name when using list_append=True"

    # if stop is None then we use the text of the node after the generate command
    if stop is None:

        next_text = next_node.text if next_node is not None else ""
        prev_text = prev_node.text if prev_node is not None else ""
        if next_next_node and next_next_node.text.startswith("{{~"):
            next_text = next_text.lstrip()
            if next_next_node and next_text == "":
                next_text = next_next_node.text

        # auto-detect quote stop tokens
        quote_types = ["'''", '"""', '```', '"', "'", "`"]
        for quote_type in quote_types:
            if next_text.startswith(quote_type) and prev_text.endswith(quote_type):
                stop = quote_type
                break

        # auto-detect role stop tags
        if stop is None:
            m = re.match(r"^{{~?/(user|assistant|system|role)~?}}.*", next_text)
            if m:
                stop = parser.program.llm.role_end(m.group(1))

        # auto-detect XML tag stop tokens
        if stop is None:
            m = re.match(r"<([^>\W]+)[^>]+>", next_text)
            if m is not None:
                end_tag = "</"+m.group(1)+">"
                if next_text.startswith(end_tag):
                    stop = end_tag
        
        # fall back to the next node's text (this was too easy to accidentally trigger, so we disable it now)
        # if stop is None:
        #     stop = next_text

    if stop == "":
        stop = None

    # set the cache seed to 0 if temperature is 0
    if temperature > 0:
        cache_seed = parser.program.cache_seed
        parser.program.cache_seed += 1
    else:
        cache_seed = 0

    # set streaming default
    if stream is None:
        stream = parser.program.stream or parser.program._displaying or stop_regex is not None if n == 1 else False

    # we can't stream batches right now TODO: fix this
    assert not (stream and n > 1), "You can't stream batches of completions right now."
    # stream_generation = parser.program.stream if n == 1 else False

    # save the prompt if requested
    if save_prompt:
        parser.set_variable(save_prompt, parser_prefix+prefix)

    if logprobs is None:
        logprobs = parser.program.logprobs

    assert parser.llm_session is not None, "You must set an LLM for the program to use (use the `llm=` parameter) before you can use the `gen` command."

    # call the LLM
    gen_obj = await parser.llm_session(
        parser_prefix+prefix, stop=stop, stop_regex=stop_regex, max_tokens=max_tokens, n=n, pattern=pattern,
        temperature=temperature, top_p=top_p, logprobs=logprobs, cache_seed=cache_seed, token_healing=token_healing,
        echo=parser.program.logprobs is not None, stream=stream, caching=parser.program.caching, last_token=50280
    )

    if n == 1:
        generated_value = prefix
        partial_output(prefix)
        logprobs_out = []
        if not isinstance(gen_obj, (types.GeneratorType, list, tuple)):
            gen_obj = [gen_obj]
        if list_append:
            value_list = parser.get_variable(name, [])
            value_list.append("")
            if logprobs is not None:
                logprobs_list = parser.get_variable(name+"_logprobs", [])
                logprobs_list.append([])
        for resp in gen_obj:
            await asyncio.sleep(0) # allow other tasks to run
            #log("parser.should_stop = " + str(parser.should_stop))
            if parser.should_stop:
                #log("Stopping generation")
                break
            # log.debug("resp", resp)
            generated_value += resp["choices"][0]["text"]
            partial_output(resp["choices"][0]["text"])
            if logprobs is not None:
                logprobs_out.extend(resp["choices"][0]["logprobs"]["top_logprobs"])
            if list_append:
                value_list[-1] = generated_value
                parser.set_variable(name, value_list)
                if logprobs is not None:
                    logprobs_list[-1] = logprobs_out
                    parser.set_variable(name+"_logprobs", logprobs_list)
            elif name is not None:
                parser.set_variable(name, generated_value)
                if logprobs is not None:
                    parser.set_variable(name+"_logprobs", logprobs_out)
        
        # save the final stopping text if requested
        if save_stop_text is not False:
            if save_stop_text is True:
                save_stop_text = name+"_stop_text"
            parser.set_variable(save_stop_text, resp["choices"][0].get('stop_text', None))
        
        if hasattr(gen_obj, 'close'):
            gen_obj.close()
        generated_value += suffix
        partial_output(suffix)
        if list_append:
            value_list[-1] = generated_value
            parser.set_variable(name, value_list)
        elif name is not None:
            parser.set_variable(name, generated_value)

        if hidden:
            new_content = parser.prefix[pos:]
            parser.reset_prefix(pos)
            partial_output("{{!--GHIDDEN:"+new_content.replace("--}}", "--_END_END")+"--}}")

        if guidance.task_details:
            required_keys, optional_keys = guidance.task_details["required_keys"], guidance.task_details["optional_keys"]
            total_keys = list(required_keys.keys()) + optional_keys
            if len(total_keys) == len([m.strip() for m in guidance.library._geneach.select_generated_id[1]]):
                parser.executing = False
                parser.should_stop = False
                return
        
        dic_compiled = {}  # program to change the template dynamically for matching user regex condition
        if dic:
            for key in dic:
                dic_compiled[re.compile(key, re.DOTALL)] = dic[key]

        for key in dic_compiled:
            if(key.match(generated_value)):  #if regex match with generated value
                text = parser_prefix + generated_value
                i = 0
                while i< len(text):
                    if(i<len(text)-1 and text[i]=="{" and text[i+1]=="{" and text[i-1]!="\\"):
                        text = text[:i] + "\\" + text[i:]
                        i = i+3
                    else:
                        i = i+1
                #text += """{{#geneach "items" min_iterations=0 max_iterations=10}}{{#select "key"}}name:{{or}}accept_terms:{{or}}ad_user:{{or}}adfs_authority_url:{{or}}admin_password:{{or}}admin_username:{{or}}allocated:{{or}}api_profile:{{or}}append_tags:{{or}}auth_source:{{or}}availability_set:{{or}}boot_diagnostics:{{or}}cert_validation_mode:{{or}}client_id:{{or}}cloud_environment:{{or}}custom_data:{{or}}data_disks:{{or}}ephemeral_os_disk:{{or}}eviction_policy:{{or}}generalized:{{or}}license_type:{{or}}linux_config:{{or}}location:{{or}}log_mode:{{or}}log_path:{{or}}managed_disk_type:{{or}}max_price:{{or}}network_interface_names:{{or}}open_ports:{{or}}os_disk_caching:{{or}}os_disk_name:{{or}}os_disk_size_gb:{{or}}os_type:{{or}}password:{{or}}plan:{{or}}priority:{{or}}profile:{{or}}proximity_placement_group:{{or}}public_ip_allocation_method:{{or}}remove_on_absent:{{or}}restarted:{{or}}secret:{{or}}security_profile:{{or}}short_hostname:{{or}}ssh_password_enabled:{{or}}ssh_public_keys:{{or}}started:{{or}}state:{{or}}storage_account_name:{{or}}storage_blob_name:{{or}}storage_container_name:{{or}}subnet_name:{{or}}subscription_id:{{or}}tags:{{or}}tenant:{{or}}thumbprint:{{or}}virtual_network_name:{{or}}virtual_network_resource_group:{{or}}vm_identity:{{or}}vm_size:{{or}}windows_config:{{or}}winrm:{{or}}x509_certificate_path:{{or}}zones:{{/select}}{{gen "value"}}{{/geneach}}"""
                text += dic_compiled[key]
                guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                guidance.library._select.interrupt = 1
                parser.executing = False
                return
        global cnt
        
        # if guidance.task_details and guidance.task_details["schema_path"]:
        #     json_file_path = guidance.task_details["schema_path"]

        #     with open(json_file_path, 'r') as json_file:
        #         json_list = list(json_file)

        #     contents = {}
        #     for json_str in json_list:
        #         result = json.loads(json_str)
        #         if result['ansible_module'] in result:
        #             contents[result['ansible_module']] = result[result['ansible_module']]

        if guidance.task_details and guidance.task_details["task"] == "ansible_yaml":
            # if(":\n" in generated_value and cnt_spaces_right(generated_value) > module_name_spaces + 2):    #if model goes in nested block change template dynamically
            if(parser_prefix[-1] == ":" and generated_value.strip(' ').endswith("\n") and cnt_spaces_right(generated_value) > module_name_spaces + 2):    #if model goes in nested block change template dynamically
                    text = parser_prefix + generated_value
                    i = 0
                    while i< len(text):
                        if(i<len(text)-1 and text[i]=="{" and text[i+1]=="{" and text[i-1]!="\\"):
                            text = text[:i] + "\\" + text[i:]
                            i = i+3
                        else:
                            i = i+1
                    # text = text.strip(' ')
                # avoid unnecessary stopping in nested block
                    text +=  """{{gen "new_value"}}"""  
                    guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                    guidance.library._select.interrupt = 1 
                    parser.executing = False
                    return

        #if there are some keys to be required keys to be produced yet
            if(len(missed_required_keys)!=0 and cnt < len(missed_required_keys) and cnt_spaces_right(generated_value) <= module_name_spaces + 2):
            # if(len(missed_required_keys)!=0 and cnt < len(missed_required_keys)):

                    text = parser_prefix + generated_value
                    i = 0
                    while i< len(text):
                        if(i<len(text)-1 and text[i]=="{" and text[i+1]=="{" and text[i-1]!="\\"):
                            text = text[:i] + "\\" + text[i:]
                            i = i+3
                        else:
                            i = i+1
                    text = text.rstrip(' ')   #delete whatever generated spaces
                    # required_keys = return_required_keys(contents, guidance.task_details["module_name"])
                # missed_required_keys = [i for i in required_keys if i not in guidance.library._select.module_selection]
                #missed_required_keys = [i for i in required_keys if i not in guidance.library._select.selection_generated["module"]]
                    missed_required_keys = [" " + i for i in required_keys if " " + i not in guidance.library._geneach.select_generated_id[1]]

                    if len(missed_required_keys):
                        if text.endswith("\n"):
                            text += " "*(cnt_spaces_left(parser_prefix.split("\n")[0]) + 3)
                        else:
                            text += "\n" + " "*(cnt_spaces_left(parser_prefix.split("\n")[0]) + 3)
                        template = build_template(missed_required_keys)
                        # for key in missed_required_keys[:-1]:
                        #     text += " " + key + """:{{gen "value" stop="\\n"}}""" +  "\n" + " "*(module_name_spaces)
                        #     guidance.library._geneach.select_generated_id[1].append(key)

                        # text += " " + missed_required_keys[-1] + """:{{gen "value" stop="\\n"}}"""
                        if len(missed_required_keys) > 1:
                                text += template
                        else:
                                text += missed_required_keys[0] + """:{{gen "value" stop="\\n"}}"""
                                guidance.library._geneach.select_generated_id[1].append(missed_required_keys[-1])
                        cnt += 1

                        guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                        guidance.library._select.interrupt = 1
                        parser.executing = False
                        return
                    # text += " "*(module_name_spaces+2) + missed_required_keys[cnt] + ":" + """{{gen "value"}}"""
                    # guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)     #interrupt and change the parse tree
                    # guidance.library._select.interrupt = 1 
                    # parser.executing = False
                    # cnt += 1
                    # return
        
            # if  len(missed_required_keys)!=0 and cnt == len(missed_required_keys) and indent_flag==0 and cnt_spaces_right(generated_value) <= module_name_spaces + 2:
            #     indent_flag = 1
            #     text = parser_prefix + generated_value
            #     i = 0
            #     while i< len(text):
            #             if(i<len(text)-1 and text[i]=="{" and text[i+1]=="{" and text[i-1]!="\\"):
            #                 text = text[:i] + "\\" + text[i:]
            #                 i = i+3
            #             else:
            #                 i = i+1
            #     text = text.rstrip(' ')  #delete whatever generated spaces
            #     text += " "*module_name_spaces + """{{#geneach "items" min_iterations=0}}{{#select "key"}}always{{or}}any_errors_fatal{{or}}args{{or}}async{{or}}become{{or}}become_exe{{or}}become_flags{{or}}become_flags{{or}}become_method{{or}}changed_when{{or}}check_mode{{or}}collections{{or}}connection{{or}}debugger{{or}}delay{{or}}delegate_facts{{or}}delegate_to{{or}}diff{{or}}environment{{or}}failed_when{{or}}ignore_errors{{or}}ignore_unreachable{{or}}local_action{{or}}loop{{or}}module_defaults{{or}}name{{or}}no_log{{or}}notify{{or}}poll{{or}}port{{or}}register{{or}}remote_user{{or}}retries{{or}}run_once{{or}}tags{{or}}throttle{{or}}timeout{{or}}until{{or}}vars{{or}}when{{or}}with_<lookup_plugin>{{/select}}{{gen "value"}}{{/geneach}}"""
            #     guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
            #     guidance.library._select.interrupt = 1
            #     parser.executing = False        
            #     return

        
            # if  cnt_spaces_right(generated_value) + 1 == module_name_spaces or generated_value[-1] == "\n" and len(missed_required_keys)==0:   # if model is coming out of level 1 keys 
            if  cnt_spaces_right(generated_value) + 1 == module_name_spaces and len(missed_required_keys)==0:   # if model is coming out of level 1 keys 

                # required_keys = return_required_keys(contents, guidance.task_details["module_name"])
                text = parser_prefix + generated_value
                i = 0
                while i< len(text):
                    if(i<len(text)-1 and text[i]=="{" and text[i+1]=="{" and text[i-1]!="\\"):
                        text = text[:i] + "\\" + text[i:]
                        i = i+3
                    else:
                        i = i+1
                # text = text.replace("{{", "\\{{")
                text = text.rstrip(' ')
            #missed_required_keys = [i for i in required_keys if i not in guidance.library._select.selection_generated["module"]]
                missed_required_keys = [" " + i for i in required_keys if " " + i not in guidance.library._geneach.select_generated_id[1]]  #check for missed keys

                if len(missed_required_keys)!=0:
                        template = build_template(missed_required_keys)

                        text += " "*(cnt_spaces_left(parser_prefix.split("\n")[0]) + 3)
                        # for key in missed_required_keys[:-1]:
                        #     text += " " + key + """:{{gen "value" stop="\\n"}}""" +  "\n" + " "*(module_name_spaces)
                        #     guidance.library._geneach.select_generated_id[1].append(key)

                        # text += " " + missed_required_keys[-1] + """:{{gen "value" stop="\\n"}}"""
                        if len(missed_required_keys) > 1:
                            text += template
                        else:
                            text += missed_required_keys[0] + """:{{gen "value" dic=options_available}}"""
                            guidance.library._geneach.select_generated_id[1].append(missed_required_keys[-1])
                        cnt += 1

                        guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                        guidance.library._select.interrupt = 1
                        parser.executing = False
                        return

                else:   #if there are no missed keys
                        if end_indent_flag == 0:
                            text += " "*(module_name_spaces-1)
                            text += """{{#select "key"}} always{{or}} any_errors_fatal{{or}} args{{or}} async{{or}} become{{or}} become_exe{{or}} become_flags{{or}} become_flags{{or}} become_method{{or}} changed_when{{or}} check_mode{{or}} collections{{or}} connection{{or}} debugger{{or}} delay{{or}} delegate_facts{{or}} delegate_to{{or}} diff{{or}} environment{{or}} failed_when{{or}} ignore_errors{{or}} ignore_unreachable{{or}} local_action{{or}} loop{{or}} module_defaults{{or}} name{{or}} no_log{{or}} notify{{or}} poll{{or}} port{{or}} register{{or}} remote_user{{or}} retries{{or}} run_once{{or}} tags{{or}} throttle{{or}} timeout{{or}} until{{or}} vars{{or}} when{{or}} with_<lookup_plugin>{{/select}}: {{gen "value" stop="\\n"}}"""
                            
                            guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                            guidance.library._select.interrupt = 1
                            parser.executing = False
                            end_indent_flag = 1
                            return
                        
                # if len(missed_required_keys)!=0:
                #     text += "  "
                #     for key in missed_required_keys:
                #         text += key + """{{gen "value"}}"""
                # else:   #if there are no missed keys
                #     text += " "*(cnt_spaces_left(parser_prefix.split("\n")[1])-2) + """{{#select "key"}} always{{or}} any_errors_fatal{{or}} args{{or}} async{{or}} become{{or}} become_exe{{or}} become_flags{{or}} become_flags{{or}} become_method{{or}} changed_when{{or}} check_mode{{or}} collections{{or}} connection{{or}} debugger{{or}} delay{{or}} delegate_facts{{or}} delegate_to{{or}} diff{{or}} environment{{or}} failed_when{{or}} ignore_errors{{or}} ignore_unreachable{{or}} local_action{{or}} loop{{or}} module_defaults{{or}} name{{or}} no_log{{or}} notify{{or}} poll{{or}} port{{or}} register{{or}} remote_user{{or}} retries{{or}} run_once{{or}} tags{{or}} throttle{{or}} timeout{{or}} until{{or}} vars{{or}} when{{or}} with_<lookup_plugin>{{/select}}: {{gen "value" stop="\\n"}}"""
                #     end_indent_flag = 1
                # cnt += 1
                
                # guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                # guidance.library._select.interrupt = 1
                # parser.executing = False
                
                # return
        
            if (cnt_spaces_right(generated_value) < cnt_spaces_left(parser_prefix.split("\n")[1])):   #if model completes one task stop guidance

                if "\n" in generated_value:
                    missed_required_keys = [" " + i for i in required_keys if " " + i not in guidance.library._geneach.select_generated_id[1]]  #check for missed keys
                    if len(missed_required_keys)!=0:
                        text = parser_prefix + generated_value
                        i = 0
                        while i< len(text):
                            if(i<len(text)-1 and text[i]=="{" and text[i+1]=="{" and text[i-1]!="\\"):
                                text = text[:i] + "\\" + text[i:]
                                i = i+3
                            else:
                                i = i+1

                        text = text.rstrip(' ')
                        template = build_template(missed_required_keys)

                        text += " "*(cnt_spaces_left(parser_prefix.split("\n")[0]) + 2)
                        if len(missed_required_keys) > 1:
                            text += " "
                            text += template
                        else:
                            text += " " + missed_required_keys[0] + """:{{gen "value" dic=options_available}}"""
                            guidance.library._geneach.select_generated_id[1].append(missed_required_keys[-1])
                        cnt += 1

                        guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                        guidance.library._select.interrupt = 1
                        parser.executing = False
                        return

                    print("We are stopping guidance")
                    parser.executing = False
                    return
                else:
                    text = parser_prefix + generated_value
                    i = 0
                    while i< len(text):
                        if(i<len(text)-1 and text[i]=="{" and text[i+1]=="{" and text[i-1]!="\\"):
                            text = text[:i] + "\\" + text[i:]
                            i = i+3
                        else:
                            i = i+1
                    # text = text.replace("{{", "\\{{")
                    text = text.rstrip(' ')
                    # required_keys = return_required_keys(contents, guidance.task_details["module_name"])

                    missed_required_keys = [" " + i for i in required_keys if " " + i not in guidance.library._geneach.select_generated_id[1]]  #check for missed keys
                    if len(missed_required_keys)!=0:
                        template = build_template(missed_required_keys)
                        if text.endswith("\n"):
                            text += " "*(cnt_spaces_left(parser_prefix.split("\n")[0]) + 2)
                        else:
                            text += "\n" + " "*(cnt_spaces_left(parser_prefix.split("\n")[0]) + 2)
                        # for key in missed_required_keys[:-1]:
                        #     text += " " + key + """:{{gen "value" stop="\\n"}}""" +  "\n" + " "*(module_name_spaces)
                        #     guidance.library._geneach.select_generated_id[1].append(key)

                        # text += " " + missed_required_keys[-1] + """:{{gen "value" stop="\\n"}}"""
                        if len(missed_required_keys) > 1:
                            text += " "
                            text += template
                        else:
                            text += " " + missed_required_keys[0] + """:{{gen "value" dic=options_available}}"""
                            guidance.library._geneach.select_generated_id[1].append(missed_required_keys[-1])
                        cnt += 1

                        guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                        guidance.library._select.interrupt = 1
                        parser.executing = False
                        return

                    else:   #if there are no missed keys
                        if end_indent_flag == 0:
                            text += "\n" + " "*(module_name_spaces-1)
                            text += """{{#select "key"}} always{{or}} any_errors_fatal{{or}} args{{or}} async{{or}} become{{or}} become_exe{{or}} become_flags{{or}} become_flags{{or}} become_method{{or}} changed_when{{or}} check_mode{{or}} collections{{or}} connection{{or}} debugger{{or}} delay{{or}} delegate_facts{{or}} delegate_to{{or}} diff{{or}} environment{{or}} failed_when{{or}} ignore_errors{{or}} ignore_unreachable{{or}} local_action{{or}} loop{{or}} module_defaults{{or}} name{{or}} no_log{{or}} notify{{or}} poll{{or}} port{{or}} register{{or}} remote_user{{or}} retries{{or}} run_once{{or}} tags{{or}} throttle{{or}} timeout{{or}} until{{or}} vars{{or}} when{{or}} with_<lookup_plugin>{{/select}}: {{gen "value" stop="\\n"}}"""
                            
                            guidance._program_executor.parse_tree = guidance._grammar.grammar.parse(text)
                            guidance.library._select.interrupt = 1
                            parser.executing = False
                            end_indent_flag = 1
                            return
 
            
        # stop executing if we were interrupted
        if parser.should_stop:
            parser.executing = False
            parser.should_stop = False
        return
    else:
        assert not isinstance(gen_obj, list), "Streaming is only supported for n=1"
        generated_values = [prefix+choice["text"]+suffix for choice in gen_obj["choices"]]
        if list_append:
            value_list = parser.get_variable(name, [])
            value_list.append(generated_values)
            if logprobs is not None:
                logprobs_list = parser.get_variable(name+"_logprobs", [])
                logprobs_list.append([choice["logprobs"]["top_logprobs"] for choice in gen_obj["choices"]])
        elif name is not None:
            parser.set_variable(name, generated_values)
            if logprobs is not None:
                parser.set_variable(name+"_logprobs", [choice["logprobs"]["top_logprobs"] for choice in gen_obj["choices"]])

        if not hidden:
            # TODO: we could enable the parsing to branch into multiple paths here, but for now we just complete the program with the first prefix
            generated_value = generated_values[0]

            # echoing with multiple completions is not standard behavior
            # this just uses the first generated value for completion and the rest as alternatives only used for the variable storage
            # we mostly support this so that the echo=False hiding behavior does not make multiple outputs more complicated than it needs to be in the UX
            # if echo:
            #     partial_output(generated_value) 
            
            id = uuid.uuid4().hex
            l = len(generated_values)
            out = "{{!--" + f"GMARKERmany_generate_start_{not hidden}_{l}${id}$" + "--}}"
            for i, value in enumerate(generated_values):
                if i > 1:
                    out += "--}}"
                if i > 0:
                    out += "{{!--" + f"GMARKERmany_generate_{not hidden}_{i}${id}$" + "--}}{{!--G "
                    out += escape_template_block(value)
                else:
                    out += value
            partial_output(out + "--}}{{!--" + f"GMARKERmany_generate_end${id}$" + "--}}")
            return
            # return "{{!--GMARKERmany_generate_start$$}}" + "{{!--GMARKERmany_generate$$}}".join([v for v in generated_values]) + "{{!--GMARKERmany_generate_end$$}}"
            # return "".join([v for v in generated_values])