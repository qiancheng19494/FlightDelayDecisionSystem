"""复用已有嵌入式画布，绘制历史描述性统计。"""
from app.widgets.chart_widget import ChartWidget

class HistoryChartWidget(ChartWidget):
    def show_history(self,trends):
        self.figure.clear()
        axes=self.figure.subplots(1,3)
        for ax,kind,title in zip(axes,["month","weekday","hour"],["月度延误率","星期延误率","计划小时延误率"]):
            items=sorted(trends.get(kind,{}).items())
            if not items:
                ax.text(.5,.5,"暂无可计算数据",ha="center",va="center",transform=ax.transAxes)
                ax.set_axis_off()
                continue
            keys=[k for k,v in items]
            if kind=="weekday": keys=[["一","二","三","四","五","六","日"][int(k)] for k in keys]
            ax.bar(range(len(items)),[v["delay_count"]/v["count"] for k,v in items],color="#397cb4")
            stride=max(1,len(keys)//6)
            ax.set_xticks(range(0,len(keys),stride),keys[::stride],rotation=25 if kind=="month" else 0)
            ax.set_ylim(0,1);ax.set_title(title)
            ax.set_ylabel("延误比例")
        self.canvas.draw_idle()

    def show_entity(self,frame,title):
        self.figure.clear()
        ax=self.figure.subplots()
        if len(frame):
            ax.plot(frame.source_date,frame.delay_rate,color="#2476ad",linewidth=1.5)
            ax.set_ylim(0,1)
            ax.set_ylabel("当日延误比例")
            ax.tick_params(axis="x",rotation=20)
        else:
            ax.text(.5,.5,"暂无实体历史",ha="center",va="center",transform=ax.transAxes)
        ax.set_title(title+" · 描述性日趋势")
        self.canvas.draw_idle()
