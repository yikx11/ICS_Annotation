<template>
    <a-locale-provider :locale='zh_CN'>
        <div id='app'>
            <!-- <a-row>
                <a-col :span='6'></a-col>
                <a-col :span='12'>
                    <MessageBox />
                    <InputPanel></InputPanel>
                </a-col>
                <a-col :span='6'></a-col>
            </a-row> -->
            <!-- <Test /> -->
            <Testl />
            <!-- <Testbrush /> -->
        </div>
    </a-locale-provider>
</template>

<script>
import zh_CN from 'ant-design-vue/lib/locale-provider/zh_CN'
import InputPanel from './components/InputPanel.vue'
import MessageBox from './components/MessageBox.vue'
// import Test from './components/Test.vue'
import Testbrush from './components/Testbrush.vue'
import Testl from './components/Testl.vue'
export default {
    name: 'app',
    data() {
        return {
            zh_CN,
        }
    },
    components: {
        MessageBox,
        InputPanel,
        Testl,
        // Test,
        // Testbrush,
    },
}
</script>

<style>
#app {
    /* font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50; */
    margin-top: 60px;
}
</style>
