#include <Python.h>
#include <iostream>
using namespace std;

static void print_str(PyObject *o)
{
    PyObject_Print(o, stdout, Py_PRINT_RAW);
}

int main(){
    Py_Initialize(); // 启动虚拟机
    if (!Py_IsInitialized())
        return -1;
    // 导入模块
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("import os");
    PyRun_SimpleString("from easydict import EasyDict");
    // PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("sys.path.append('/Users/puyuan/code/LightZeroLightZero/zoo/board_games/tictactoe/envs/')");
    PyRun_SimpleString("print('current work path', os.getcwd())");
    PyRun_SimpleString("from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv");
    PyRun_SimpleString("cfg = EasyDict(dict(prob_random_agent=0, prob_expert_agent=0, battle_mode='one_player_mode', agent_vs_human=False,))");
    PyRun_SimpleString("env = TicTacToeEnv(EasyDict(cfg))");
    PyRun_SimpleString("env.reset()");

    // PyObject* pModule = PyImport_ImportModule("test123");
    // PyObject* pClass = PyImport_ImportModule("tictactoe_env");
    PyRun_SimpleString("print(env.board)");
    // int action = 1;
    // PyRun_SimpleString("env.step(action)");
    PyObject* pEnv = PyImport_ImportModule("env");
    // PyObject* pEnv = PyObject_GetAttrString("env");

    PyObject *action = Py_BuildValue("i", 1);
    PyObject_CallMethod(pEnv, "step", "i", action);
    PyObject* pEnvStates = PyObject_GetAttrString(pEnv, "board");
    // print_str(pEnvStates);

    // PyObject *pReturn = PyObject_CallMethod(pEnv, "have_winner", NULL, NULL);
    // int x,y;
    // PyArg_Parse(pReturn, "i", &x);
    // PyArg_Parse(pReturn, "i", &y);
    // std::cout<<x<<std::endl;

    // PyRun_SimpleString("print(env.board)");

    // if (!pClass) {
    //     printf("Cant open python file!/n");
    //     PyErr_Print();
    //     return -1;
    // }

    // // 模块的字典列表
    // PyObject* pEnvDict = PyModule_GetDict(pClass);
    // if (!pEnvDict) {
    //     printf("Cant find dictionary./n");
    //     return -1;
    // }

    // // 演示构造一个Python对象，并调用Class的方法
    // // 获取Second类
    // PyObject* pEnv = PyDict_GetItemString(pEnvDict, "TicTacToeEnv");
    // if (!pEnv) {
    //     printf("Cant find second class./n");
    //     return -1;
    // }
    // //构造Second的实例
    // PyObject* pInstanceEnv = PyObject_CallFunction(pEnv, pCfg, pCfg);
    // if (!pInstanceEnv) {
    //     printf("Cant create second instance./n");
    //     PyErr_Print();
    //     return -1;
    // }
    // //构造Person的实例
    // PyObject* pInstancePerson = PyObject_CallFunction(pClassPerson, NULL, NULL);
    // if (!pInstancePerson) {
    //     printf("Cant find person instance./n");
    //     return -1;
    // }
    // //把person实例传入second的invoke方法
    // PyObject_CallMethod(pInstanceSecond, "invoke", "O", pInstancePerson);
    // //释放
    // Py_DECREF(pInstanceSecond);
    // Py_DECREF(pInstancePerson);
    // Py_DECREF(pClassSecond);
    // Py_DECREF(pClassPerson);
    // Py_DECREF(pModule);
    Py_Finalize(); // 关闭虚拟机
    std::cout<<"hello"<<std::endl;
    return 0;
}