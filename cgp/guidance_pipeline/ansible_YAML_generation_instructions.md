To run the evaluation script on indivivdual example run the testing.py file by providing the arguments prompt, module name and schema path.

Example python3 testing.py <prompt> <module name> <schema path> <task type>

To run YAML generation on complete data run run_NL2YAML.py file. 
This data will be fetched from path "/dccstor/rhassistant/ansible_wisdom/sbert_information_retrieval/sbert_mnrl_data/module_docs_13524.json".

Generated YAML will be stored in YAML.txt file in main guidance folder. 

Generated examples:

For the prompt:
- name: install and configure apache web server
  hosts: managed
  become: true
  tasks:
    - name: install httpd package
      ansible.builtin.yum:
        name: httpd
        state: present
<|sepoftext|>    - name: start and enable apache
      ansible.builtin.service:

Full Generated YAML
- name: install and configure apache web server
  hosts: managed
  become: true
  tasks:
    - name: install httpd package
      ansible.builtin.yum:
        name: httpd
        state: present
<|sepoftext|>    - name: start and enable apache
      ansible.builtin.service:
        name: httpd
        state: started
        enabled: true

For the prompt:

- name: Create an Azure VM
  hosts: localhost
  gather_facts: false
  collections:
    - azure.azcollection
    - community.general    
  vars: 



    servers_to_be_deployed: ['test-postgresql-1','test-postgresql-2']
  tasks:
    - name: Create Azure VMs
      azure.azcollection.azure_rm_virtualmachine:

Full Generated YAML:

- name: Create an Azure VM
  hosts: localhost
  gather_facts: false
  collections:
    - azure.azcollection
    - community.general    
  vars: 



    servers_to_be_deployed: ['test-postgresql-1','test-postgresql-2']
  tasks:
    - name: Create Azure VMs
      azure.azcollection.azure_rm_virtualmachine:
        resource_group: "{{ item.resource_group }}"
        name: "{{ item.name }}"
        vm_size: "{{ item.vm_size }}"
        admin_username: "{{ item.admin_username }}"
        public_ip_allocation_method: "{{ item.public_ip_allocation_method }}"
        ssh_password_enabled: false
        ssh_public_keys:
          - path: /home/adminUser/.ssh/authorized_keys
            key_data: < insert your ssh public key here... >
        network_interface_names: [test-network-interface-1]
        storage_container_name: "{{ item.storage_container_name }}"
        storage_blob_name: "{{ item.storage_blob_name }}"
        image:
          offer: CentOS
          publisher: OpenLogic
          sku: "7.1"
          version: latest
      register: vm
      loop: "{{ Servers }}"

Openshift prompt: 

kind: "ImageStream"
apiVersion: "v1"
metadata:
  name: "ruby"
  creationTimestamp: null
spec:
  dockerImageRepository: "registry.redhat.io/rhscl/ruby-26-rhel7"
  tags:

Full generated YAML:

kind: "ImageStream"
apiVersion: "v1"
metadata:
  name: "ruby"
  creationTimestamp: null
spec:
  dockerImageRepository: "registry.redhat.io/rhscl/ruby-26-rhel7"
  tags:
    - privileged




