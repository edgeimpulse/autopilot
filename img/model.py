
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

def to_Relu( name, s_filer=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +"""
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        xlabel={{" ","dummy"}},
        zlabel="""+ str(s_filer) +""",
        fill=\FcReluColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Dropout( name, s_filer=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +"""
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        xlabel={{" ","dummy"}},
        zlabel="""+ str(s_filer) +""",
        fill=\PoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# Conv
def to_ConvRelu( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" ", zlabel="" ):
    return r"""
\tikzstyle{zlabel}=[pos=1.0,text width=14*\z,text centered,sloped]
\pic[shift={"""+ offset +"""}] at """+ to +"""
    {RightBandedBox={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(zlabel) +""",
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_input( './autopilot.jpg', width=7, height=4 ),

    to_ConvRelu("conv1", "", 16, offset="(0,0,0)", to="(0,0,0)", height=30, depth=30, width=3, caption="conv", zlabel="I/2" ),

    to_ConvRelu("conv2", "", 16, offset="(1.5,0,0)", to="(conv1-east)", height=20, depth=20, width=3, caption="conv", zlabel="I/4" ),
    to_connection( "conv1", "conv2"),

    to_ConvRelu("conv3", "", 32, offset="(1.5,0,0)", to="(conv2-east)", height=15, depth=15, width=6, caption="conv", zlabel="I/8" ),
    to_connection( "conv2", "conv3"),

    to_ConvRelu("conv4", "", 32, offset="(1.5,0,0)", to="(conv3-east)", height=15, depth=15, width=6, caption="conv", zlabel="I/8" ),
    to_connection( "conv3", "conv4"),

    to_Pool("pool1", offset="(0.5,0,0)", to="(conv4-east)", height=10, depth=10, caption="maxpool"),

    to_Dropout("drop1", "" ,"(2.5,0,0)", "(pool1-east)", caption="dropout", depth=32 ),
    to_connection("pool1", "drop1"),

    to_Relu("soft1", 32 ,"(1.5,0,0)", "(drop1-east)", caption="relu", depth=32  ),
    to_connection( "drop1", "soft1"),

    to_Dropout("drop2", "" ,"(1.5,0,0)", "(soft1-east)", caption="dropout", depth=16 ),
    to_connection( "soft1", "drop2"),

    to_Relu("soft2", 16 ,"(1.5,0,0)", "(drop2-east)", caption="relu", depth=16  ),
    to_connection( "drop2", "soft2"),

    to_SoftMax("soft3", 7 ,"(1.5,0,0)", "(soft2-east)", caption="softmax", depth=7  ),
    to_connection( "soft2", "soft3"),

    to_end()

    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

